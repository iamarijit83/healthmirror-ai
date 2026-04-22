import os
from PIL import Image

BASE_PATH = "data/fer2013"
REQUIRED_CLASSES = {"angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"}

def check_split(split):
    print(f"\n🔍 Checking {split.upper()} dataset...")
    
    path = os.path.join(BASE_PATH, split)

    if not os.path.exists(path):
        print(f"❌ Missing folder: {path}")
        return None

    classes = set(os.listdir(path))
    
    print(f"📁 Found classes: {classes}")

    # Check missing classes
    missing = REQUIRED_CLASSES - classes
    if missing:
        print(f"❌ Missing classes: {missing}")
    else:
        print("✅ All required classes present")

    class_counts = {}

    for cls in classes:
        cls_path = os.path.join(path, cls)

        if not os.path.isdir(cls_path):
            print(f"⚠️ {cls} is not a folder")
            continue

        files = os.listdir(cls_path)
        count = 0
        corrupted = 0

        for file in files:
            file_path = os.path.join(cls_path, file)

            try:
                img = Image.open(file_path)
                img.verify()  # check corruption
                count += 1
            except:
                corrupted += 1

        class_counts[cls] = count

        print(f"   {cls}: {count} images | ⚠️ corrupted: {corrupted}")

        if count == 0:
            print(f"❌ ERROR: {cls} folder is empty!")

    return classes, class_counts


# -------------------------
# RUN CHECKS
# -------------------------
train_classes, train_counts = check_split("train")
test_classes, test_counts = check_split("test")

# -------------------------
# CONSISTENCY CHECK
# -------------------------
print("\n🔁 Checking consistency between TRAIN and TEST...")

if train_classes == test_classes:
    print("✅ Train/Test classes match")
else:
    print("❌ Class mismatch between train and test")
    print("Train:", train_classes)
    print("Test :", test_classes)

print("\n📊 Dataset Summary:")
total_train = sum(train_counts.values()) if train_counts else 0
total_test = sum(test_counts.values()) if test_counts else 0

print(f"Total Train Images: {total_train}")
print(f"Total Test Images : {total_test}")

print("\n✅ Dataset verification COMPLETE")