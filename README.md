# Road Detection (道路辨識)

## 一、需求
* **功能**：於道路區域疊加 半透明紅色標記之影像。
* **限制**：含直線道路、街景及天空之影像。解析度**：以 **1280x720** 為基準。
* **界面**：以 png檔案輸入。

## 二、分析
本專案專注於路面之語義分割與標註，結合多重特徵過濾以排除非道路干擾：

| 特徵類別 | 技術說明 | 偵測目的 |
| :--- | :--- | :--- |
| **色彩特徵** | HSV 瀝青灰與標線黃/白色彩遮罩 | 識別道路基本色調與標線邊界 |
| **空間幾何** | **Tight ROI** 梯形投影遮罩 (收窄底邊) | 強力排除兩側斜坡、人行道與招牌 |
| **紋理特徵** | Sobel 梯度平滑度計算 | 排除碎石、樹木等高紋理之非道路雜訊 |
| **區域連貫性** | SLIC 超像素分群與 **Seed Color** 約束 | 強化邊界貼合度，依據種子色排除干擾物 |

## 三、設計
### 1. 系統流程 (Pipeline)
```mermaid
graph TD
    A[輸入影像] -->|np.ndarray, uint8, 720x1280x3| B[色彩與對比度增強 CLAHE]
    B -->|np.ndarray, uint8| C[梯度平滑度/幾何梯形過濾]
    C -->|np.ndarray, int32| D[SLIC 超像素分群]
    D -->|np.ndarray, uint8| E[超像素投票與色彩種子約束]
    E -->|np.ndarray, uint8| F[GrabCut 邊界精煉優化]
    F -->|np.ndarray, uint8| G[最大連通區篩選]
    G -->|np.ndarray, uint8| H[結果疊加標註]
    H -->|np.ndarray, uint8, 720x1280x3| I[輸出最終影像]
```

### 2. Pipeline 演算法細節
1. **影像預處理**：採用 CLAHE (`cv2.createCLAHE`) 提升暗部路面對比度。
2. **幾何與紋理約束**：建構動態 **Tight ROI** (收窄底邊之梯形) 與 Sobel 梯度遮罩，有效解決影像兩側斜坡誤判問題。
3. **SLIC 分群**：分割影像為 500 個超像素區塊 (`skimage.segmentation.slic`)，增加邊界細節。
4. **多重顏色遮罩**：鎖定瀝青灰與車道標線黃/白區域 (`cv2.inRange`)。
5. **色彩種子約束**：計算超像素與底部中心「**路面種子 (Seed Color)**」的歐幾里得距離，作為排除顏色相近但色調不同物體之關鍵門檻。
6. **GrabCut 精煉**：以投票結果為基礎進行 3 次 GrabCut 迭代 (`cv2.grabCut`)，自動優化切割邊界。
7. **最大連通區**：定位影像中面積最大且符合幾何特徵的物件作為主道路 (`cv2.connectedComponentsWithStats`)。
8. **結果疊加**：影像透明度疊加顯示，標註偵測範圍 (`cv2.addWeighted`)。

## 四、驗證
### 階段性對比 (Stages Comparison)
![image](process_comparison/Figure_1.png)
*Fig 1.1 處理流程：原始影像 ➜ 綜合特徵遮罩 ➜ 最終優化結果 (Image 1)*

![image](process_comparison/Figure_2.png)
*Fig 1.2 處理流程：原始影像 ➜ 綜合特徵遮罩 ➜ 最終優化結果 (Image 2)*

![image](process_comparison/Figure_3.png)
*Fig 1.3 處理流程：原始影像 ➜ 綜合特徵遮罩 ➜ 最終優化結果 (Image 3)*

![image](process_comparison/Figure_4.png)
*Fig 1.4 處理流程：原始影像 ➜ 綜合特徵遮罩 ➜ 最終優化結果 (Image 4)*

![image](process_comparison/Figure_5.png)
*Fig 1.5 處理流程：原始影像 ➜ 綜合特徵遮罩 ➜ 最終優化結果 (Image 5)*

![image](process_comparison/Figure_6.png)
*Fig 1.6 處理流程：原始影像 ➜ 綜合特徵遮罩 ➜ 最終優化結果 (Image 6)*

---

## 五、參考資料 (References)
1. **Achanta, R., et al.** "SLIC Superpixels Compared to State-of-the-art Superpixel Methods." *IEEE PAMI*, 2012.
2. **Rother, C., et al.** "GrabCut: Interactive Foreground Extraction using Iterated Graph Cuts." *ACM SIGGRAPH*, 2004.
