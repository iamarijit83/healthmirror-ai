import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v2
from torch import nn, optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.tensorboard import SummaryWriter

# -------------------------
# CONFIG
# -------------------------
BATCH_SIZE = 32
LR = 0.0003
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_PATH = "data/fer2013/train"
TEST_PATH = "data/fer2013/test"

CHECKPOINT_DIR = "training/emotion/checkpoints"
LOG_DIR = "training/emotion/logs"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# -------------------------
# TENSORBOARD
# -------------------------
writer = SummaryWriter(LOG_DIR)

# -------------------------
# TRANSFORMS
# -------------------------
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# -------------------------
# DATA
# -------------------------
train_data = ImageFolder(TRAIN_PATH, transform=train_transform)
test_data = ImageFolder(TEST_PATH, transform=test_transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

NUM_CLASSES = len(train_data.classes)

print("Classes:", train_data.classes)

# -------------------------
# MODEL (MobileNetV2)
# -------------------------
model = mobilenet_v2(pretrained=True)

# Freeze base layers
for param in model.features.parameters():
    param.requires_grad = False

# Replace classifier
model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)

model = model.to(DEVICE)

# -------------------------
# LOSS + OPTIMIZER
# -------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# -------------------------
# TRAINING LOOP
# -------------------------
best_acc = 0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    preds = []
    labels_list = []

    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    for images, labels in tqdm(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        preds.extend(predicted.cpu().numpy())
        labels_list.extend(labels.cpu().numpy())

    train_acc = accuracy_score(labels_list, preds)
    train_loss = running_loss / len(train_loader)

    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Accuracy/train", train_acc, epoch)

    # -------------------------
    # VALIDATION
    # -------------------------
    model.eval()
    val_preds = []
    val_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            val_preds.extend(predicted.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_acc = accuracy_score(val_labels, val_preds)
    val_f1 = f1_score(val_labels, val_preds, average='weighted')

    print(f"Val Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

    writer.add_scalar("Accuracy/val", val_acc, epoch)
    writer.add_scalar("F1/val", val_f1, epoch)

    # -------------------------
    # CHECKPOINTS
    # -------------------------
    torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/epoch_{epoch}.pth")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/best_model.pth")
        print("✅ Best model saved!")

writer.close()

print("Training Completed!")