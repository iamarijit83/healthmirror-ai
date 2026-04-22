import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter

# -------------------------
# CONFIG
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
LR = 0.0001
EPOCHS = 40

CHECKPOINT_PATH = "models/checkpoint.pth"
BEST_MODEL_PATH = "models/best_emotion_model.pth"

print(f"Using device: {DEVICE}")

# -------------------------
# TRANSFORMS (Improved)
# -------------------------
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((48, 48)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

# -------------------------
# DATASET
# -------------------------
train_data = ImageFolder("data/fer2013/train", transform=train_transform)
val_data = ImageFolder("data/fer2013/test", transform=val_transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

# -------------------------
# CLASS WEIGHTS (IMPORTANT)
# -------------------------
labels = [label for _, label in train_data.samples]
class_counts = Counter(labels)
total = sum(class_counts.values())

class_weights = []
for i in range(len(class_counts)):
    class_weights.append(total / class_counts[i])

class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

print("Class weights:", class_weights)

# -------------------------
# MODEL
# -------------------------
model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

model.classifier[1] = torch.nn.Linear(model.last_channel, 7)

# 🔥 FULL UNFREEZE (Phase 2)
for param in model.parameters():
    param.requires_grad = True

model = model.to(DEVICE)

# -------------------------
# LOSS + OPTIMIZER
# -------------------------
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', patience=3, factor=0.5
)

# -------------------------
# LOAD CHECKPOINT
# -------------------------
start_epoch = 0
best_acc = 0

if os.path.exists(CHECKPOINT_PATH):
    print("🔁 Loading checkpoint...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = checkpoint["epoch"] + 1
    best_acc = checkpoint["best_acc"]

# -------------------------
# TRAIN LOOP
# -------------------------
for epoch in range(start_epoch, EPOCHS):

    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()

        # 🔥 Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

        optimizer.step()

        train_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total

    # -------------------------
    # VALIDATION
    # -------------------------
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")

    print(f"Train Acc: {train_acc:.4f}")
    print(f"Val Acc: {val_acc:.4f}, F1: {f1:.4f}")

    scheduler.step(val_acc)

    # -------------------------
    # SAVE BEST MODEL
    # -------------------------
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print("✅ Best model saved")

    # -------------------------
    # SAVE CHECKPOINT
    # -------------------------
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_acc": best_acc
    }, CHECKPOINT_PATH)

print("\n🔥 Training Complete")