import cv2
import numpy as np
import os
from skimage.segmentation import slic
from skimage.util import img_as_float

# 讀取圖片
input_dir = "input_images"
image_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".jpg")])
if not image_files:
    print("找不到圖片，請確認 input_images 資料夾內有 .jpg 檔案。")
    exit()

cv2.namedWindow('Tuning Parameter GUI', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Tuning Parameter GUI', 600, 800)

def nothing(x):
    pass

# 建立滑桿 (Trackbars)
cv2.createTrackbar('Image_Idx', 'Tuning Parameter GUI', 0, len(image_files)-1, nothing)
cv2.createTrackbar('SLIC_N', 'Tuning Parameter GUI', 500, 1000, nothing)
cv2.createTrackbar('SLIC_Comp', 'Tuning Parameter GUI', 15, 50, nothing)

# HSV 範圍調整
cv2.createTrackbar('H_min', 'Tuning Parameter GUI', 0, 180, nothing)
cv2.createTrackbar('S_min', 'Tuning Parameter GUI', 0, 255, nothing)
cv2.createTrackbar('V_min', 'Tuning Parameter GUI', 40, 255, nothing)
cv2.createTrackbar('H_max', 'Tuning Parameter GUI', 180, 180, nothing)
cv2.createTrackbar('S_max', 'Tuning Parameter GUI', 45, 255, nothing)
cv2.createTrackbar('V_max', 'Tuning Parameter GUI', 170, 255, nothing)

# ROI 與投票門檻
cv2.createTrackbar('ROI_Top%', 'Tuning Parameter GUI', 50, 100, nothing)
cv2.createTrackbar('Vote_Ratio%', 'Tuning Parameter GUI', 30, 100, nothing)

last_idx = -1
last_slic_n = -1
last_slic_comp = -1
segments = None
image = None
rgb = None
hsv = None

print("========================================")
print("GUI 調參工具已啟動！")
print("請在跳出的視窗中拉動滑桿調整參數。")
print("為保證流暢度，預覽圖已縮小為 640x360。")
print("按下 'ESC' 鍵即可關閉視窗退出。")
print("========================================")

while True:
    idx = cv2.getTrackbarPos('Image_Idx', 'Tuning Parameter GUI')
    slic_n = cv2.getTrackbarPos('SLIC_N', 'Tuning Parameter GUI')
    slic_comp = cv2.getTrackbarPos('SLIC_Comp', 'Tuning Parameter GUI')
    
    # 避免 SLIC 參數為 0 導致報錯
    if slic_n < 10: slic_n = 10
    if slic_comp < 1: slic_comp = 1

    h_min = cv2.getTrackbarPos('H_min', 'Tuning Parameter GUI')
    s_min = cv2.getTrackbarPos('S_min', 'Tuning Parameter GUI')
    v_min = cv2.getTrackbarPos('V_min', 'Tuning Parameter GUI')
    h_max = cv2.getTrackbarPos('H_max', 'Tuning Parameter GUI')
    s_max = cv2.getTrackbarPos('S_max', 'Tuning Parameter GUI')
    v_max = cv2.getTrackbarPos('V_max', 'Tuning Parameter GUI')
    
    roi_top = cv2.getTrackbarPos('ROI_Top%', 'Tuning Parameter GUI') / 100.0
    vote_ratio = cv2.getTrackbarPos('Vote_Ratio%', 'Tuning Parameter GUI') / 100.0

    # 當切換圖片時，重新讀取圖片
    if idx != last_idx:
        img_path = os.path.join(input_dir, image_files[idx])
        original_image = cv2.imread(img_path)
        # 縮小圖片一半以加快即時運算速度 (640x360)
        image = cv2.resize(original_image, (640, 360))
        
        # 這裡不套用 CLAHE，因為我們想直接透過 HSV 找到最真實的路面顏色
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        last_idx = idx
        last_slic_n = -1 # 強制重新計算 SLIC

    # 當 SLIC 參數改變時，重新計算 SLIC 分群 (這步較耗時)
    if slic_n != last_slic_n or slic_comp != last_slic_comp:
        img_float = img_as_float(rgb)
        segments = slic(img_float, n_segments=slic_n, compactness=slic_comp, sigma=1, start_label=1, channel_axis=2)
        last_slic_n = slic_n
        last_slic_comp = slic_comp

    h, w = image.shape[:2]
    
    # 1. HSV 遮罩 (原始過濾)
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    color_mask = cv2.inRange(hsv, lower, upper)

    # 2. ROI 空間遮罩 (切除上方天空與遠景)
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    roi_mask[int(h * roi_top):, :] = 255
    color_mask = cv2.bitwise_and(color_mask, roi_mask)

    # 3. 超像素投票
    mask = np.zeros((h, w), dtype=np.uint8)
    for seg_val in np.unique(segments):
        region = (segments == seg_val)
        ratio = np.mean(color_mask[region] > 0)
        if ratio > vote_ratio:
            mask[region] = 255

    # 4. 形態學清理
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 5. 最大連通區篩選
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        sorted_indices = np.argsort(areas)[::-1]
        best_label = 1
        for sid in sorted_indices:
            label_idx = sid + 1
            if stats[label_idx, cv2.CC_STAT_TOP] + stats[label_idx, cv2.CC_STAT_HEIGHT]/2 > h * 0.5:
                best_label = label_idx
                break
        final_mask = (labels == best_label).astype(np.uint8) * 255
    else:
        final_mask = mask

    # 6. 結果疊加顯示
    overlay = image.copy()
    overlay[final_mask > 0] = [0, 0, 255] # 疊加紅色
    output = cv2.addWeighted(overlay, 0.4, image, 0.6, 0)
    
    # 排版顯示：左上(原圖 HSV遮罩結果) 右上(最終輸出)
    color_mask_bgr = cv2.cvtColor(color_mask, cv2.COLOR_GRAY2BGR)
    
    # 寫上文字標示
    cv2.putText(color_mask_bgr, "1. HSV + ROI Mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(output, "2. Final Overlay", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 水平合併顯示
    combined = np.hstack((color_mask_bgr, output))
    cv2.imshow('Tuning Parameter GUI', combined)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 27: # 按下 ESC 鍵退出
        # 離開前在終端機印出最後調整的參數
        print("\n--- 您最後選擇的參數 ---")
        print(f"SLIC_N (分群數): {slic_n}")
        print(f"SLIC_Comp (緊密度): {slic_comp}")
        print(f"HSV Lower: [{h_min}, {s_min}, {v_min}]")
        print(f"HSV Upper: [{h_max}, {s_max}, {v_max}]")
        print(f"ROI_Top%: {roi_top}")
        print(f"Vote_Ratio (投票門檻): {vote_ratio}")
        print("------------------------\n")
        break

cv2.destroyAllWindows()
