import os
import time

import torch
import torch.nn as nn
import streamlit as st

from PIL import Image
from torchvision import transforms, models


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Industrial Defect Detector",
    page_icon="🔍",
    layout="centered"
)


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "models/defect_resnet18.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    classes = checkpoint["classes"]

    model = models.resnet18(
        weights=None
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        len(classes)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)
    model.eval()

    return model, classes


model, classes = load_model()


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# HEADER
# ============================================================

st.title(
    "🔍 Industrial Surface Defect Detector"
)

st.markdown(
    """
**Deep Learning powered defect classification**

Upload a surface-defect image and the ResNet18
model will classify it into one of six defect categories.
"""
)

st.divider()


# ============================================================
# MODEL INFORMATION
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Model",
        "ResNet18"
    )

with col2:
    st.metric(
        "Validation Accuracy",
        "97.19%"
    )

with col3:
    st.metric(
        "Classes",
        "6"
    )


st.divider()


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a surface defect image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    st.divider()

    image_tensor = transform(
        image
    ).unsqueeze(0).to(DEVICE)

    with st.spinner(
        "Analyzing defect..."
    ):

        start_time = time.perf_counter()

        with torch.no_grad():

            outputs = model(
                image_tensor
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

        inference_time = (
            time.perf_counter()
            - start_time
        )

    confidence, predicted_idx = torch.max(
        probabilities,
        dim=1
    )

    predicted_class = classes[
        predicted_idx.item()
    ]

    confidence = confidence.item()

    # Top 3
    top_probs, top_indices = torch.topk(
        probabilities,
        min(3, len(classes))
    )


    # ========================================================
    # RESULT
    # ========================================================

    st.subheader(
        "Prediction"
    )

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        st.success(
            f"**{predicted_class.upper()}**"
        )

    with result_col2:

        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )


    st.metric(
        "Inference Time",
        f"{inference_time * 1000:.2f} ms"
    )


    # ========================================================
    # TOP PREDICTIONS
    # ========================================================

    st.subheader(
        "Top Predictions"
    )

    for probability, index in zip(
        top_probs[0],
        top_indices[0]
    ):

        class_name = classes[
            index.item()
        ]

        score = probability.item()

        st.write(
            f"**{class_name}** — "
            f"{score * 100:.2f}%"
        )

        st.progress(
            float(score)
        )


    # ========================================================
    # MODEL DETAILS
    # ========================================================

    with st.expander(
        "Model Details"
    ):

        st.write(
            "**Architecture:** ResNet18"
        )

        st.write(
            "**Framework:** PyTorch"
        )

        st.write(
            "**Input Size:** 224 × 224"
        )

        st.write(
            "**Task:** 6-class surface defect classification"
        )

        st.write(
            f"**Device:** {DEVICE}"
        )