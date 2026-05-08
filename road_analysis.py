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
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

for i, img_file in enumerate(image_files):
    image_path = os.path.join(input_dir, img_file)
    image = cv2.imread(image_path)
    if image is None:
        continue

    h, w = image.shape[:2]
    enhanced = optimize_image(image)
    
    # 1. 取得道路色彩種子 (底部中央區域)
    seed_y1, seed_y2 = int(h*0.85), int(h*0.95)
    seed_x1, seed_x2 = int(w*0.45), int(w*0.55)
    seed_area = enhanced[seed_y1:seed_y2, seed_x1:seed_x2]
    avg_seed_color = np.mean(seed_area, axis=(0, 1))

    # 2. 梯度平滑度
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
    abs_sobel = np.absolute(sobel)
    smooth_mask = np.where(abs_sobel < 60, 255, 0).astype(np.uint8)

    # 3. 幾何梯形 ROI (針對斜坡問題收窄底邊)
    roi_poly = np.array([
        [int(w*0.15), h],           # 收窄底邊從 0.02 -> 0.15
        [int(w*0.45), int(h*0.42)], 
        [int(w*0.55), int(h*0.42)], 
        [int(w*0.85), h]            # 收窄底邊從 0.98 -> 0.85
    ], dtype=np.int32)
    geom_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(geom_mask, [roi_poly], 255)

    # 4. 多重顏色遮罩
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    lower_gray = np.array([0, 0, 30])
    upper_gray = np.array([180, 100, 180])
    mask_gray = cv2.inRange(hsv, lower_gray, upper_gray)
    
    lower_yellow = np.array([15, 30, 80])
    upper_yellow = np.array([40, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    road_seed_mask = cv2.bitwise_or(mask_gray, mask_yellow)
    
    # 5. SLIC
    rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    segments = slic(img_as_float(rgb), n_segments=500, compactness=10, channel_axis=2)

    # 綜合過濾 (顏色 + 幾何 + 平滑)
    combined_ref_mask = cv2.bitwise_and(road_seed_mask, geom_mask)
    combined_ref_mask = cv2.bitwise_and(combined_ref_mask, smooth_mask)

    # 6. 超像素投票 (加入色彩種子距離檢查)
    mask = np.zeros((h, w), dtype=np.uint8)
    for seg_val in np.unique(segments):
        region = (segments == seg_val)
        
        # 計算與種子顏色的歐幾里得距離
        seg_color = np.mean(enhanced[region], axis=0)
        color_dist = np.linalg.norm(seg_color - avg_seed_color)
        
        # 投票門檻
        if np.mean(combined_ref_mask[region] > 0) > 0.25 and color_dist < 60: 
            mask[region] = 255

    # 7. 橫向連通強化
    bridge_kernel = np.ones((5, 25), np.uint8) 
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, bridge_kernel)

    # 8. GrabCut & Final CC
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    gc_mask = np.where((mask > 0), 3, 2).astype('uint8')
    if np.any(gc_mask == 3):
        cv2.grabCut(image, gc_mask, None, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_MASK)
        mask = np.where((gc_mask == 1) | (gc_mask == 3), 255, 0).astype('uint8')

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        final_mask = (labels == largest).astype(np.uint8) * 255
    else:
        final_mask = mask

    overlay = image.copy()
    overlay[final_mask > 0] = [0, 0, 255]
    output = cv2.addWeighted(overlay, 0.4, image, 0.6, 0)
    cv2.imwrite(os.path.join(output_dir, f"road_final_slic_{img_file}"), output)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1); plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)); plt.title("Original")
    plt.subplot(1, 3, 2); plt.imshow(combined_ref_mask, cmap="gray"); plt.title("Seed Color & Tight ROI")
    plt.subplot(1, 3, 3); plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB)); plt.title("Final Result")
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, f"Figure_{i+1}.png"))
    plt.close()

print("完成：斜坡排除優化已執行。")
