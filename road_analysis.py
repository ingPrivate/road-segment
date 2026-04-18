import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.util import img_as_float

# =========================
# 建立目錄
# =========================
output_dir = "output_results"
os.makedirs(output_dir, exist_ok=True)
comparison_dir = "process_comparison"
os.makedirs(comparison_dir, exist_ok=True)

input_dir = "input_images"
image_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".jpg")])

for i, img_file in enumerate(image_files):
    image_path = os.path.join(input_dir, img_file)
    image = cv2.imread(image_path)
    if image is None:
        continue

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # =========================
    # SLIC Superpixel
    # =========================
    img_float = img_as_float(rgb)

    segments = slic(
        img_float,
        n_segments=300,
        compactness=10,
        sigma=1,
        start_label=1,
        channel_axis=2
    )

    # =========================
    # Road color mask (HSV)
    # =========================
    lower = np.array([0, 0, 50])
    upper = np.array([180, 80, 220])
    color_mask = cv2.inRange(hsv, lower, upper)

    # =========================
    # Superpixel voting
    # =========================
    mask = np.zeros(gray.shape, dtype=np.uint8)

    for seg_val in np.unique(segments):
        region = (segments == seg_val)
        ratio = np.mean(color_mask[region] > 0)
        if ratio > 0.5:   # threshold 可調
            mask[region] = 255

    # =========================
    # Morphology clean
    # =========================
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # =========================
    # 最大連通區
    # =========================
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        final_mask = (labels == largest).astype(np.uint8) * 255
    else:
        final_mask = mask

    # =========================
    # overlay
    # =========================
    overlay = image.copy()
    overlay[final_mask > 0] = [0, 0, 255]
    output = cv2.addWeighted(overlay, 0.5, image, 0.5, 0)

    # =========================
    # 儲存
    # =========================
    output_filename = f"road_final_slic_{img_file}"
    output_path = os.path.join(output_dir, output_filename)
    cv2.imwrite(output_path, output)

    # 儲存第一張圖的階段對比圖
    if i == 0:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 3, 1)
        plt.title("Original")
        plt.imshow(rgb)
        plt.subplot(1, 3, 2)
        plt.title("SLIC Segments")
        plt.imshow(segments, cmap="nipy_spectral")
        plt.subplot(1, 3, 3)
        plt.title("Final Road Mask")
        plt.imshow(final_mask, cmap="gray")
        plt.tight_layout()
        plt.savefig(os.path.join(comparison_dir, "Figure_1.png"))
        plt.close()

print("完成：SLIC road segmentation 已處理所有圖片，並儲存對比圖。")
