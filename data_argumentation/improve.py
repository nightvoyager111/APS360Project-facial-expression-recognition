import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

# === CONFIG ===
SOURCE_DIR = 'RAF-DB/test'       # Set your original dataset directory
OUTPUT_DIR = 'RAF-DB_clean/test' # Set your destination for enhanced images
ENHANCE_CONTRAST = True           # Toggle contrast enhancement (CLAHE)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Image transforms (basic)
base_transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

# CLAHE contrast enhancer
def apply_clahe(img_bgr):
    img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
    img_clahe = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img_clahe

# Process and save each image
for root, dirs, files in os.walk(SOURCE_DIR):
    rel_path = os.path.relpath(root, SOURCE_DIR)
    target_dir = os.path.join(OUTPUT_DIR, rel_path)
    os.makedirs(target_dir, exist_ok=True)

    for file in tqdm(files, desc=f"Processing {rel_path}"):
        if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        src_path = os.path.join(root, file)
        dst_path = os.path.join(target_dir, file)

        try:
            # Open image using PIL then convert to OpenCV
            img_pil = Image.open(src_path).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

            # Apply CLAHE if enabled
            if ENHANCE_CONTRAST:
                img_bgr = apply_clahe(img_bgr)

            # Resize and convert back to PIL for saving
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_pil_out = Image.fromarray(img_rgb).resize((48, 48))

            # Save to output path
            img_pil_out.save(dst_path)
        except Exception as e:
            print(f"Error processing {src_path}: {e}")
