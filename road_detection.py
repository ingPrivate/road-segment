import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
import os

# 建立輸出目錄
output_dir = "output_results"
os.makedirs(output_dir, exist_ok=True)

# 讀取圖片目錄
input_dir = "input_images"
image_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".jpg")])

for img_file in image_files:
    image_path = os.path.join(input_dir, img_file)
    image = cv2.imread(image_path)
    if image is None:
        continue

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 直方圖均衡化（Gray Image）
    equalized_gray = cv2.equalizeHist(gray_image)

    # 設定道路顏色的 HSV 範圍
    lower_gray = np.array([0, 0, 20])
    upper_gray = np.array([180, 45, 200])
    road_mask = cv2.inRange(hsv_image, lower_gray, upper_gray)

    # 計算 LBP 並僅保留道路區域
    def compute_lbp(image, mask, P=8, R=2):
        lbp = local_binary_pattern(image, P, R, method='uniform')
        lbp = lbp * (mask // 255)  # 只保留道路區域的 LBP 值
        return lbp

    lbp_image = compute_lbp(equalized_gray, road_mask)

    # 對 LBP 圖像進行二值化
    _, mask = cv2.threshold(lbp_image.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 尋找輪廓並保留最大區塊
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        continue
        
    # 找到最大輪廓
    max_contour = max(contours, key=cv2.contourArea)

    # 創建一個空白遮罩來保存最大區塊
    filtered_mask = np.zeros_like(mask)
    cv2.drawContours(filtered_mask, [max_contour], -1, 255, thickness=cv2.FILLED)

    # 將道路部分覆蓋紅色
    overlay = image.copy()
    alpha = 0.5
    overlay[filtered_mask > 0] = [0, 0, 255]  # 紅色
    output = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    # 儲存結果
    output_filename = f"road_final_lbp_{img_file}"
    output_path = os.path.join(output_dir, output_filename)
    cv2.imwrite(output_path, output)

print("完成：LBP road detection 已處理所有圖片")
