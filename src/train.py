import os
import xml.etree.ElementTree as ET
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = r"C:\Users\soham\industrial-defect-detection\data\NEU-DET"

TRAIN_DIR = os.path.join(DATASET_ROOT, "train")
VAL_DIR = os.path.join(DATASET_ROOT, "validation")

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("INDUSTRIAL SURFACE DEFECT DETECTION")
print("=" * 60)
print(f"Device: {DEVICE}")

if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Running on CPU")


# ============================================================
# DATASET PARSER
# ============================================================

def parse_annotations(split_dir):
    """
    Parse XML annotations and match each annotation
    to its image inside the class-specific image folder.
    """

    annotation_dir = os.path.join(split_dir, "annotations")
    image_root = os.path.join(split_dir, "images")

    records = []

    for xml_file in os.listdir(annotation_dir):

        if not xml_file.lower().endswith(".xml"):
            continue

        xml_path = os.path.join(annotation_dir, xml_file)

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            filename_node = root.find("filename")

            if filename_node is None:
                continue

            filename = filename_node.text.strip()

            for obj in root.findall("object"):

                name_node = obj.find("name")

                if name_node is None:
                    continue

                label = name_node.text.lower().strip()

                # ------------------------------------------------
                # Find image inside class-specific directory
                # Example:
                # images/scratches/scratches_9.jpg
                # ------------------------------------------------

                image_path = os.path.join(
                    image_root,
                    label,
                    filename
                )

                # Fallback: search all image subdirectories
                if not os.path.exists(image_path):

                    for class_folder in os.listdir(image_root):

                        candidate = os.path.join(
                            image_root,
                            class_folder,
                            filename
                        )

                        if os.path.exists(candidate):
                            image_path = candidate
                            break

                if not os.path.exists(image_path):
                    continue

                # ------------------------------------------------
                # Bounding box
                # ------------------------------------------------

                bbox = obj.find("bndbox")

                if bbox is None:
                    continue

                xmin = int(float(bbox.find("xmin").text))
                ymin = int(float(bbox.find("ymin").text))
                xmax = int(float(bbox.find("xmax").text))
                ymax = int(float(bbox.find("ymax").text))

                # Validate bounding box
                if xmax <= xmin or ymax <= ymin:
                    continue

                records.append({
                    "image": image_path,
                    "bbox": (xmin, ymin, xmax, ymax),
                    "label": label
                })

        except Exception as e:
            print(f"Warning: could not parse {xml_file}: {e}")

    return records

print("\nLoading annotations...")

train_records = parse_annotations(TRAIN_DIR)
val_records = parse_annotations(VAL_DIR)

print(f"Training defect regions: {len(train_records)}")
print(f"Validation defect regions: {len(val_records)}")

# ============================================================
# LABEL ENCODING
# ============================================================

classes = sorted(set(r["label"] for r in train_records))
class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}

print("\nClasses:")
for idx, cls in idx_to_class.items():
    print(f"{idx}: {cls}")

NUM_CLASSES = len(classes)


# ============================================================
# DATASET
# ============================================================

class DefectDataset(Dataset):

    def __init__(self, records, class_to_idx, transform=None):

        self.records = records
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):

        record = self.records[idx]

        image = Image.open(record["image"]).convert("RGB")

        xmin, ymin, xmax, ymax = record["bbox"]

        # Prevent invalid coordinates
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(image.width, xmax)
        ymax = min(image.height, ymax)

        # Crop annotated defect
        image = image.crop((xmin, ymin, xmax, ymax))

        label = self.class_to_idx[record["label"]]

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================
# IMAGE TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(10),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


train_dataset = DefectDataset(
    train_records,
    class_to_idx,
    train_transform
)

val_dataset = DefectDataset(
    val_records,
    class_to_idx,
    val_transform
)


# ============================================================
# HANDLE CLASS IMBALANCE
# ============================================================

train_labels = [
    class_to_idx[r["label"]]
    for r in train_records
]

class_counts = Counter(train_labels)

print("\nTraining distribution:")

for idx, count in sorted(class_counts.items()):
    print(f"{idx_to_class[idx]:20s}: {count}")


weights = []

for label in train_labels:
    weights.append(
        1.0 / class_counts[label]
    )

sampler = WeightedRandomSampler(
    weights,
    num_samples=len(weights),
    replacement=True
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# MODEL
# ============================================================

print("\nLoading pretrained ResNet18...")

weights = ResNet18_Weights.DEFAULT

model = models.resnet18(weights=weights)

# Freeze the pretrained backbone
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the final ResNet block for domain-specific fine-tuning
for param in model.layer4.parameters():
    param.requires_grad = True

# Replace classification head
model.fc = nn.Linear(
    model.fc.in_features,
    NUM_CLASSES
)

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

# Smaller learning rate for fine-tuning
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4
)


# ============================================================
# TRAINING
# ============================================================

train_losses = []
val_losses = []

train_accuracies = []
val_accuracies = []

best_val_accuracy = 0.0


print("\nStarting training...")
print("=" * 60)


for epoch in range(EPOCHS):

    # -------------------------
    # TRAIN
    # -------------------------

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS}"
    )

    for images, labels in progress:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

        progress.set_postfix(
            loss=loss.item()
        )

    train_loss = running_loss / len(train_loader)

    train_accuracy = 100 * correct / total


    # -------------------------
    # VALIDATION
    # -------------------------

    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()

            all_predictions.extend(
                predicted.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )


    val_loss /= len(val_loader)

    val_accuracy = (
        100 * val_correct / val_total
    )


    train_losses.append(train_loss)
    val_losses.append(val_loss)

    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)


    print(
        f"\nEpoch {epoch + 1}/{EPOCHS}"
    )

    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.2f}%"
    )

    print(
        f"Val Loss: {val_loss:.4f} | "
        f"Val Accuracy: {val_accuracy:.2f}%"
    )


    # Save best model
    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save({
            "model_state_dict": model.state_dict(),
            "classes": classes,
            "class_to_idx": class_to_idx
        }, os.path.join(
            MODEL_DIR,
            "defect_resnet18.pth"
        ))

        print("✓ Best model saved")


# ============================================================
# LOAD BEST CHECKPOINT
# ============================================================

best_checkpoint = torch.load(
    os.path.join(MODEL_DIR, "defect_resnet18.pth"),
    map_location=DEVICE
)

model.load_state_dict(
    best_checkpoint["model_state_dict"]
)

model.eval()

print("\nBest checkpoint loaded for final evaluation.")

# ============================================================
# FINAL EVALUATION
# ============================================================

print("\n")
print("=" * 60)
print("FINAL EVALUATION")
print("=" * 60)

print(
    f"\nBest Validation Accuracy: "
    f"{best_val_accuracy:.2f}%"
)


print("\nClassification Report:")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=classes,
        digits=4
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

plt.figure(figsize=(9, 7))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=classes,
    yticklabels=classes
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.title(
    "Industrial Defect Classification - Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300
)

plt.close()


# ============================================================
# TRAINING CURVES
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    train_losses,
    label="Training Loss"
)

plt.plot(
    val_losses,
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title("Training vs Validation Loss")

plt.legend()

plt.tight_layout()

plt.savefig(
    "training_loss.png",
    dpi=300
)

plt.close()


plt.figure(figsize=(8, 5))

plt.plot(
    train_accuracies,
    label="Training Accuracy"
)

plt.plot(
    val_accuracies,
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")

plt.title(
    "Training vs Validation Accuracy"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "training_accuracy.png",
    dpi=300
)

plt.close()


print("\nTraining complete.")

print("\nGenerated files:")

print("models/defect_resnet18.pth")
print("confusion_matrix.png")
print("training_loss.png")
print("training_accuracy.png")