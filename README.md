# 顧客購買行為預測與推薦系統 (Customer Purchase Prediction & Recommendation)

本專案旨在使用機器學習模型預測顧客是否會購買特定商品，並根據預測機率產出推薦清單（Top 12 商品）。專案涵蓋了從資料讀取、模型訓練（Decision Tree, Random Forest, XGBoost）、特徵重要性分析到最終提交檔案生成的完整流程。

## 目錄

  - [環境需求](https://www.google.com/search?q=%23%E7%92%B0%E5%A2%83%E9%9C%80%E6%B1%82)
  - [資料結構](https://www.google.com/search?q=%23%E8%B3%87%E6%96%99%E7%B5%90%E6%A7%8B)
  - [模型架構與策略](https://www.google.com/search?q=%23%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%A7%8B%E8%88%87%E7%AD%96%E7%95%A5)
  - [模型訓練與參數](https://www.google.com/search?q=%23%E6%A8%A1%E5%9E%8B%E8%A8%93%E7%B7%B4%E8%88%87%E5%8F%83%E6%95%B8)
  - [評估結果](https://www.google.com/search?q=%23%E8%A9%95%E4%BC%B0%E7%B5%90%E6%9E%9C)
  - [輸出檔案說明](https://www.google.com/search?q=%23%E8%BC%B8%E5%87%BA%E6%AA%94%E6%A1%88%E8%AA%AA%E6%98%8E)

## 環境需求

本專案使用 Python 進行開發，主要依賴以下套件：

```bash
pip install scikit-learn xgboost pyarrow fastparquet numpy pandas
```

*注意：XGBoost 模型設定中啟用了 GPU 加速 (`device='cuda'`)，請確保您的環境支援 NVIDIA CUDA，否則請調整參數改為 CPU 運行。*

## 資料結構

程式預期讀取 Parquet 格式的資料集：

需自行前往下載並加以處理放置資料集於指定路徑 (`./storage/`)。

``` text

label                           int8
days_last_buy                  int16
trend                        float16
user_cat_affinity            float16
user_color_affinity          float16
price_diff                   float16
FN                           float16
Active                       float16
club_member_status              int8
fashion_news_frequency          int8
age_group                       int8
product_type_name              int16
product_group_name              int8
colour_group_name               int8
index_group_name                int8
section_name                    int8
graphical_appearance_name       int8
price_group                     int8
season_score                 float16

```

  - **訓練集**: `./storage/train_set.parquet`
  - **驗證集**: `./storage/valid_set_part_*.parquet` (由 28 個部分組成並合併)
  - **測試集**: `./storage/test_set_part_*.parquet` (由 28 個部分組成並合併)

**主要特徵 (Features):**

  - **用戶行為**: `days_last_buy` (距離上次購買天數), `trend`, `active`, `club_member_status`
  - **商品特徵**: `product_type_name`, `product_group_name`, `colour_group_name`, `price_diff`
  - **其他**: `season_score`, `fashion_news_frequency` 等

**標籤 (Target):**

  - `label`: 1 (購買), 0 (未購買)

## 模型架構與策略

本專案實作並比較了三種分類模型：

1.  **決策樹 (Decision Tree)**
2.  **隨機森林 (Random Forest)**
3.  **極度梯度提升樹 (XGBoost)** - *（主要採用的最終模型）*

### 關鍵訓練策略 (針對 XGBoost)

為了處理資料不平衡與時間序列特性，採取了以下特殊處理：

  * **加權樣本 (Sample Weighting)**:
    針對近期購買行為進行加權，邏輯為：若標籤為 `1` 且 `days_last_buy < 50`，權重設為 **35.0**，其餘為 1.0。這讓模型更重視近期的購買偏好。
  * **類別不平衡處理**:
    決策樹與隨機森林使用了 `class_weight='balanced'`。
  * **GPU 加速**:
    XGBoost 使用 `tree_method='hist'` 搭配 `device='cuda'` 進行高效運算。

## 模型訓練與參數

### XGBoost 核心參數

  - `n_estimators`: 10,000 (搭配 Early Stopping: 100 rounds)
  - `learning_rate`: 0.01
  - `max_depth`: 7
  - `min_child_weight`: 100
  - `subsample`: 0.8
  - `colsample_bytree`: 0.6
  - `scale_pos_weight`: 1

## 評估結果

模型在驗證集上的表現重點如下：

  * **AUC 分數**: 70.46%
  * **Accuracy**: 96.51% (由於負樣本極多，準確率參考價值較低)
  * **最佳 F1-Score 門檻值**: 0.5

### 特徵重要性 (Top 5)

1.  `days_last_buy` (0.3349) - 最具決定性的特徵
2.  `trend` (0.2112)
3.  `price_diff` (0.0646)
4.  `product_group_name` (0.0521)
5.  `section_name` (0.0512)

### 商業指標 (Lift Analysis)

若針對預測分數前 **5%** 的客戶進行行銷：

  - **行銷效率提升 (Lift)**: **6.47 倍**
  - **名單精準度**: 0.35% (相比基準平均 0.05%)

## 輸出檔案說明

程式執行後會產生以下檔案：

1.  **`modal_train_result.xlsx`**:
    記錄歷次模型訓練的指標（Accuracy, Precision, Recall, F1, AUC）。
2.  **`feature.xlsx`**:
    記錄歷次訓練的特徵重要性排名。
3.  **`submission_top12.xlsx`**:
    最終提交檔案。格式為每個 `customer_id` 對應預測機率最高的 12 個 `article_id` (以空白分隔)。