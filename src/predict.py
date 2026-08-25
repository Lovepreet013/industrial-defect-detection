import os
import time
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "models/defect_resnet18.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

classes = checkpoint["classes"]
class_to_idx = checkpoint["class_to_idx"]

num_classes = len(classes)


# ============================================================
# LOAD MODEL
# ============================================================

model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    num_classes
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)

model.eval()


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    image_tensor = transform(
        image
    ).unsqueeze(0).to(DEVICE)

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
        time.perf_counter() - start_time
    )

    confidence, predicted_idx = torch.max(
        probabilities,
        dim=1
    )

    predicted_class = classes[
        predicted_idx.item()
    ]

    confidence = confidence.item()

    # Top 3 predictions
    top_probs, top_indices = torch.topk(
        probabilities,
        min(3, len(classes))
    )

    top_predictions = []

    for prob, idx in zip(
        top_probs[0],
        top_indices[0]
    ):

        top_predictions.append({
            "class": classes[idx.item()],
            "confidence": prob.item()
        })

    return {
        "prediction": predicted_class,
        "confidence": confidence,
        "inference_time_ms": inference_time * 1000,
        "top_predictions": top_predictions
    }


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

if __name__ == "__main__":

    import glob

    image_root = r"data\NEU-DET\validation\images"

    # Find all validation images automatically
    image_files = []

    for extension in ("*.jpg", "*.jpeg", "*.png"):
        image_files.extend(
            glob.glob(
                os.path.join(
                    image_root,
                    "*",
                    extension
                )
            )
        )

    if not image_files:
        print("\nError: No validation images found.")
        exit()

    print("\nAvailable defect classes:")

    for i, class_name in enumerate(classes):
        print(f"{i + 1}. {class_name}")

    choice = input(
        "\nSelect a class number (1-6): "
    ).strip()

    try:
        class_index = int(choice) - 1
        selected_class = classes[class_index]

    except (ValueError, IndexError):
        print("\nInvalid selection.")
        exit()

    # Find images belonging to selected class
    selected_images = [
        path
        for path in image_files
        if os.path.basename(
            os.path.dirname(path)
        ) == selected_class
    ]

    if not selected_images:
        print(
            f"\nNo images found for {selected_class}."
        )
        exit()

    # Use the first available image
    image_path = selected_images[0]

    print("\nTesting image:")
    print(image_path)

    result = predict(image_path)

    print("\n" + "=" * 55)
    print("INDUSTRIAL DEFECT CLASSIFICATION")
    print("=" * 55)

    print(
        f"\nActual Class: {selected_class}"
    )

    print(
        f"Prediction:   {result['prediction']}"
    )

    print(
        f"Confidence:   "
        f"{result['confidence'] * 100:.2f}%"
    )

    print(
        f"Inference:    "
        f"{result['inference_time_ms']:.2f} ms"
    )

    print("\nTop Predictions:")

    for item in result["top_predictions"]:

        print(
            f"  {item['class']:20s}"
            f"{item['confidence'] * 100:.2f}%"
        )

    print("=" * 55)