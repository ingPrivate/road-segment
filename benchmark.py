import cv2
import numpy as np
import os

def calculate_metrics(mask):
    if np.sum(mask) == 0:
        return 0, 0, 0
    
    # 1. 面積占比
    area = np.sum(mask > 0)
    total_area = mask.shape[0] * mask.shape[1]
    area_ratio = (area / total_area) * 100
    
    # 2. 邊緣平滑度 (Solidity)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return area_ratio, 0, 0
    
    cnt = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0
    
    # 3. 邊緣鋸齒度 (Perimeter / Area)
    perimeter = cv2.arcLength(cnt, True)
    jaggedness = perimeter / area if area > 0 else 0
    
    return area_ratio, solidity, jaggedness

results = []
input_dir = "input_images"
output_dir = "output_results"
image_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".jpg")])

print(f"{'Image':<12} | {'Method':<10} | {'Area%':<7} | {'Solidity':<8} | {'Jagged'}")
print("-" * 60)

for img_file in image_files:
    # 只讀取 SLIC+GC 的結果
    slic_res_path = os.path.join(output_dir, f"road_final_slic_{img_file}")
    
    method, path = "SLIC+GC", slic_res_path
    res_img = cv2.imread(path)
    if res_img is not None:
        hsv = cv2.cvtColor(res_img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
        
        a, s, j = calculate_metrics(mask)
        print(f"{img_file:<12} | {method:<10} | {a:>6.2f}% | {s:>8.3f} | {j:>6.5f}")
        results.append((img_file, method, a, s, j))

# 總結統計
print("\n--- 統計平均值 ---")
m = "SLIC+GC"
m_data = [r for r in results if r[1] == m]
if m_data:
    avg_a = np.mean([r[2] for r in m_data])
    avg_s = np.mean([r[3] for r in m_data])
    avg_j = np.mean([r[4] for r in m_data])
    print(f"[{m}] Avg Area: {avg_a:.2f}%, Avg Solidity: {avg_s:.3f}, Avg Jagged: {avg_j:.5f}")
