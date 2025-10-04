# LGBM_TEest1
本版本為第三代 (`v3`) 優化流程，整合時間行為、金額異常、通道多樣性等進階特徵，並使用 **LightGBM GPU 加速訓練**，提供高效且穩定的可疑帳戶預測模型。
## AI Cup_LGBM/
├── dataset/ # 官方提供資料
│ ├── acct_transaction.csv # 交易紀錄
│ ├── acct_alert.csv # 警示帳戶（標記 1）
│ └── acct_predict.csv # 要預測的帳號清單
│
├── feature_data_v3/ # 特徵工程輸出
│ └── account_features.csv
│
├── train_data_v3/ # 欠採樣後訓練集與模型
│ ├── train_data.csv
│ └── lightgbm_model.pkl
│
├── results_v3/ # 預測輸出
│ ├── predict_full.csv # 含預測機率
│ └── predict_for_submit.csv # 上傳用 (alert_key, predict)
│
├── v3/
│ ├── build_features_v3_behavior.py # 特徵建構 (含時間行為分析)
│ ├── prepare_trainset_balanced_v3.py # 欠採樣產生訓練集
│ ├── train_model_LGBM_v3.py # GPU 訓練模型
│ └── predict_accounts_v3_gpu_final.py # 預測與輸出提交檔



## 🚀 執行流程

### 第一步：特徵建構  
**檔案：** `build_features_v3_behavior.py`


主要特徵包含：

💰 金額統計（平均、最大、最小、標準差）

🔄 自轉交易比例

⏰ 時間行為特徵（早/晚交易、活躍時段）

🌐 通道多樣性 (channel_type)

🔗 對手帳號多樣性

❓ 未知類別比例 (UNK 比例)

⏱️ 首筆 / 末筆交易時間 (first_txn, last_txn)


第二步：資料平衡處理
檔案： prepare_trainset_balanced_v3.py


讀取警示帳戶 (acct_alert.csv)

從特徵表中取出警示帳號 + 隨機抽樣同量正常帳號

形成 1:1 平衡訓練集

輸出： train_data_v3/train_data.csv


第三步：訓練 LightGBM 模型 (GPU 版本)
檔案： train_model_LGBM_v3.py

特點：

使用 device_type = "gpu" 啟用 GPU 加速

早停條件：early_stopping_rounds=100

評估指標：binary_logloss

自動輸出模型績效（F1, Precision, Recall, Confusion Matrix）

輸出：

模型檔案：train_data_v3/lightgbm_model.pkl


第四步：帳號預測與提交檔產生
檔案： predict_accounts_v3_gpu_final.py

功能：

自動對齊特徵欄位

將 first_txn / last_txn 轉換為 UNIX 秒數

使用 GPU 進行批次預測

自動驗證上傳格式（確保 predict ∈ {0,1}）

輸出：

檔案	用途
results_v3/predict_full.csv	含帳號、機率與標籤
results_v3/predict_for_submit.csv	AI CUP 上傳格式 ✅


⚙️ GPU 環境需求
請確認系統安裝：


📈 模型表現 (v3 範例結果)
指標	值
F1 Score	0.94
Precision	0.95
Recall	0.93
Accuracy	0.94
最佳 Threshold	0.60
Confusion Matrix	[[192, 9], [13, 188]]

🏁 最終提交檔格式
results_v3/predict_for_submit.csv

alert_key	predict
001a...f29	0
002b...d64	1
...	...


👨‍💻 使用說明摘要
bash
複製程式碼
# 建立特徵
python v3/build_features_v3_behavior.py

欠採樣建立訓練集
python v3/prepare_trainset_balanced_v3.py

使用 GPU 訓練模型
python v3/train_model_LGBM_v3.py

使用 GPU 預測提交
python v3/predict_accounts_v3_gpu_final.py
