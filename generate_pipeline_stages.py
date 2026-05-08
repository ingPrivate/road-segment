import cv2
import numpy as np
import os
from skimage.segmentation import slic, mark_boundaries
from skimage.util import img_as_float

# 建立目錄
stage_dir = "pipeline_stages"
os.makedirs(stage_dir, exist_ok=True)

def optimize_image(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

# 讀取 road_01.jpg
image_path = "input_images/road_01.jpg"
image = cv2.imread(image_path)
h, w = image.shape[:2]

# Stage 1: Input Image
cv2.imwrite(f"{stage_dir}/stage1_input.jpg", image)

# Stage 2: CLAHE (Contrast Enhancement)
enhanced = optimize_image(image)
cv2.imwrite(f"{stage_dir}/stage2_clahe.jpg", enhanced)

# Stage 3: Geometry & Texture Filtering (Combined Mask)
# Seed Color (for next stage but calculated here)
seed_y1, seed_y2 = int(h*0.85), int(h*0.95)
seed_x1, seed_x2 = int(w*0.45), int(w*0.55)
seed_area = enhanced[seed_y1:seed_y2, seed_x1:seed_x2]
avg_seed_color = np.mean(seed_area, axis=(0, 1))

gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
abs_sobel = np.absolute(sobel)
smooth_mask = np.where(abs_sobel < 60, 255, 0).astype(np.uint8)

roi_poly = np.array([[int(w*0.15), h], [int(w*0.45), int(h*0.42)], [int(w*0.55), int(h*0.42)], [int(w*0.85), h]], dtype=np.int32)
geom_mask = np.zeros((h, w), dtype=np.uint8)
cv2.fillPoly(geom_mask, [roi_poly], 255)

hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
mask_gray = cv2.inRange(hsv, np.array([0, 0, 30]), np.array([180, 100, 180]))
mask_yellow = cv2.inRange(hsv, np.array([15, 30, 80]), np.array([40, 255, 255]))
road_seed_mask = cv2.bitwise_or(mask_gray, mask_yellow)

combined_ref_mask = cv2.bitwise_and(cv2.bitwise_and(road_seed_mask, geom_mask), smooth_mask)
cv2.imwrite(f"{stage_dir}/stage3_combined_mask.jpg", combined_ref_mask)

# Stage 4: SLIC Superpixels
rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
segments = slic(img_as_float(rgb), n_segments=500, compactness=10, channel_axis=2)
slic_view = mark_boundaries(img_as_float(rgb), segments)
cv2.imwrite(f"{stage_dir}/stage4_slic.jpg", (slic_view * 255).astype(np.uint8)[:,:,::-1])

# Stage 5: Voting & Seed Color Constraint
mask = np.zeros((h, w), dtype=np.uint8)
for seg_val in np.unique(segments):
    region = (segments == seg_val)
    seg_color = np.mean(enhanced[region], axis=0)
    color_dist = np.linalg.norm(seg_color - avg_seed_color)
    if np.mean(combined_ref_mask[region] > 0) > 0.25 and color_dist < 60: 
        mask[region] = 255
cv2.imwrite(f"{stage_dir}/stage5_voting.jpg", mask)

# Stage 6: GrabCut Refinement
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
gc_mask = np.where((mask > 0), 3, 2).astype('uint8')
if np.any(gc_mask == 3):
    cv2.grabCut(image, gc_mask, None, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_MASK)
    mask = np.where((gc_mask == 1) | (gc_mask == 3), 255, 0).astype('uint8')
cv2.imwrite(f"{stage_dir}/stage6_grabcut.jpg", mask)

# Stage 7: Max Connected Component
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
if num_labels > 1:
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    final_mask = (labels == largest).astype(np.uint8) * 255
else:
    final_mask = mask
cv2.imwrite(f"{stage_dir}/stage7_final_mask.jpg", final_mask)

# Stage 8: Result Overlay
overlay = image.copy()
overlay[final_mask > 0] = [0, 0, 255]
output = cv2.addWeighted(overlay, 0.4, image, 0.6, 0)
cv2.imwrite(f"{stage_dir}/stage8_overlay.jpg", output)

print("Pipeline stages generated successfully.")
