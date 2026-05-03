import os
import random
from pathlib import Path
import cv2
import albumentations as A

# 📁 TARGET FOLDER
TARGET_DIR = Path("server/app/ml/dataset/Train_Augmented_Balanced/5")

# 🎯 TARGET COUNT
TARGET_COUNT = 100

# 📦 Augmentation pipeline
transform = A.Compose([
    A.Rotate(limit=20, p=0.7),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.7),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussianBlur(p=0.3),
])

# 📊 Get all existing images
images = list(TARGET_DIR.glob("*.png"))

print(f"Current images: {len(images)}")

if len(images) >= TARGET_COUNT:
    print("Already reached target count.")
    exit()

# 🔁 Generate more images
counter = 0

while len(images) < TARGET_COUNT:
    img_path = random.choice(images)
    image = cv2.imread(str(img_path))

    if image is None:
        continue

    augmented = transform(image=image)["image"]

    # Generate new filename
    base_name = img_path.stem.split("_original")[0].split("_aug")[0]
    new_name = f"{base_name}_aug_extra_{counter}.png"

    save_path = TARGET_DIR / new_name

    cv2.imwrite(str(save_path), augmented)

    images.append(save_path)
    counter += 1

    print(f"Created: {new_name}")

print(f"✅ Done! Total images: {len(images)}")