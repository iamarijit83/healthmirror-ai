import os

folders = [
    "data/emotion/train",
    "data/emotion/test",
    "training/emotion",
    "training/emotion/logs",
    "training/emotion/checkpoints"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

print("✅ Folder structure created successfully!")