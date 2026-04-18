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

def optimize_image(image):
    # 使用 CLAHE 提升對比度 (調降 clipLimit 避免暗部樹木過度曝光)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

for i, img_file in enumerate(image_files):
    image_path = os.path.join(input_dir, img_file)
    image = cv2.imread(image_path)
    if image is None:
        continue

    # 1. 預處理：增強對比度
    enhanced = optimize_image(image)
    h, w = image.shape[:2]
    rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)

    # 2. SLIC Superpixel (調整緊密度與分群數)
    img_float = img_as_float(rgb)
    segments = slic(
        img_float,
        n_segments=600,
        compactness=15, # 稍微調高緊密度，避免區塊過度延伸到樹木
        sigma=1,
        start_label=1,
        channel_axis=2
    )

    # 3. 多重顏色遮罩 (廣義路面顏色)
    # 瀝青灰色 (收緊飽和度與提高最低亮度，嚴格排除暗處樹木)
    lower_gray = np.array([0, 0, 40])
    upper_gray = np.array([180, 120, 170])
    mask_gray = cv2.inRange(hsv, lower_gray, upper_gray)

    # 黃色標線 (避免路面被分割)
    lower_yellow = np.array([15, 40, 40])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    color_mask = cv2.bitwise_or(mask_gray, mask_yellow)

    # 4. 空間 ROI (動態調整，保留下半部)
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    roi_mask[int(h * 0.50):, :] = 255 # 根據截圖將 ROI 起點改為 50%
    color_mask = cv2.bitwise_and(color_mask, roi_mask)

    # 5. 超像素投票
    mask = np.zeros((h, w), dtype=np.uint8)
    for seg_val in np.unique(segments):
        region = (segments == seg_val)
        ratio = np.mean(color_mask[region] > 0)
        if ratio > 0.3: # 調低門檻以捕捉更多路面
            mask[region] = 255

    # 6. 形態學清理
    kernel = np.ones((11, 11), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 7. 最大連通區篩選
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num_labels > 1:
        # 找出面積最大且位於下半部的區塊
        areas = stats[1:, cv2.CC_STAT_AREA]
        sorted_indices = np.argsort(areas)[::-1]
        
        best_label = 1
        for idx in sorted_indices:
            label_idx = idx + 1
            # 檢查中心點是否在下半部
            if stats[label_idx, cv2.CC_STAT_TOP] + stats[label_idx, cv2.CC_STAT_HEIGHT]/2 > h*0.5:
                best_label = label_idx
                break
        
        final_mask = (labels == best_label).astype(np.uint8) * 255
    else:
        final_mask = mask

    # 8. Overlay & Save
    overlay = image.copy()
    overlay[final_mask > 0] = [0, 0, 255]
    output = cv2.addWeighted(overlay, 0.4, image, 0.6, 0)
    
    output_path = os.path.join(output_dir, f"road_final_slic_{img_file}")
    cv2.imwrite(output_path, output)

    # 產出對比圖
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title(f"Original ({img_file})")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.subplot(1, 3, 2)
    plt.title("SLIC Segments (w/ CLAHE)")
    plt.imshow(segments, cmap="nipy_spectral")
    plt.subplot(1, 3, 3)
    plt.title("Improved Road Mask")
    plt.imshow(final_mask, cmap="gray")
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, f"Figure_{i+1}.png"))
    plt.close()

print("完成：SLIC 改良演算法已執行，包含 CLAHE 與多重遮罩邏輯。")
