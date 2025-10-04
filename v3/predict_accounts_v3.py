# ===============================================================
#  predict_accounts_v3_gpu_final.py  (AI CUP 驗證版)
# ===============================================================

import pandas as pd
import joblib
import os
import numpy as np

# ==========================
# 1️⃣ 路徑設定
# ==========================
feature_file = "feature_data_v3/account_features.csv"
predict_file = "dataset/acct_predict.csv"
model_file = "train_data_v3/lightgbm_model.pkl"
output_dir = "results_v3"
os.makedirs(output_dir, exist_ok=True)

output_full = os.path.join(output_dir, "predict_full.csv")
submit_file = os.path.join(output_dir, "predict_for_submit.csv")

# ==========================
# 2️⃣ 載入資料與模型
# ==========================
print(f"📂 載入特徵表：{feature_file}")
features = pd.read_csv(feature_file)

print(f"📂 載入預測帳戶：{predict_file}")
predict_accts = pd.read_csv(predict_file)

print(f"📦 載入 LightGBM 模型：{model_file}")
model = joblib.load(model_file)

# ==========================
# 3️⃣ 自動偵測帳號欄位
# ==========================
acct_col = None
for col in predict_accts.columns:
    if col.strip().lower() in ["acct", "acct_id"]:
        acct_col = col
        break

if acct_col is None:
    raise ValueError("❌ acct_predict.csv 中沒有找到帳號欄位 (acct 或 acct_id)")

predict_accts.rename(columns={acct_col: "acct_id"}, inplace=True)

# ==========================
# 4️⃣ 篩出要預測的帳號
# ==========================
target_accts = set(predict_accts["acct_id"])
predict_features = features[features["acct_id"].isin(target_accts)].copy()

if len(predict_features) == 0:
    print("⚠️ 沒有匹配帳號，嘗試去除空白後重新比對 ...")
    features["acct_id"] = features["acct_id"].astype(str).str.strip()
    predict_accts["acct_id"] = predict_accts["acct_id"].astype(str).str.strip()
    predict_features = features[features["acct_id"].isin(predict_accts["acct_id"])].copy()

if len(predict_features) == 0:
    raise ValueError("❌ 沒有找到任何匹配的帳號特徵，請確認帳號格式是否一致。")

print(f"🎯 找到 {len(predict_features)} 筆要預測的帳號特徵")

# ==========================
# 5️⃣ 轉換時間欄位
# ==========================
for col in ["first_txn", "last_txn"]:
    if col in predict_features.columns:
        try:
            predict_features[col] = pd.to_datetime(predict_features[col], errors="coerce")
            predict_features[col] = (predict_features[col] - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
            print(f"🕒 已將欄位 {col} 轉換為 UNIX 秒數格式")
        except Exception as e:
            print(f"⚠️ 欄位 {col} 轉換失敗：{e}")

# ==========================
# 6️⃣ 對齊模型特徵欄位
# ==========================
print("🚀 使用 GPU 進行預測中 ...")

train_features = model.feature_name()
missing_cols = [c for c in train_features if c not in predict_features.columns]
extra_cols = [c for c in predict_features.columns if c not in train_features]

if missing_cols:
    print(f"⚠️ 缺少欄位（模型曾用過但現在沒有）：{missing_cols}")
if extra_cols:
    print(f"⚠️ 多出欄位（現在有但模型沒用過）：{extra_cols}")

X_pred = predict_features.reindex(columns=train_features, fill_value=0)

# ==========================
# 7️⃣ GPU 預測
# ==========================
y_pred_prob = model.predict(X_pred, num_iteration=model.best_iteration)
threshold = 0.5
predict_features["predict_prob"] = y_pred_prob
predict_features["predict_label"] = (y_pred_prob >= threshold).astype(int)

# ==========================
# 8️⃣ 儲存結果 (含格式驗證)
# ==========================
submission = predict_features[["acct_id", "predict_label", "predict_prob"]].copy()
submission = submission.rename(columns={
    "acct_id": "alert_key",
    "predict_label": "predict",
    "predict_prob": "probability"
})

# 強制轉為 int 0/1 格式
submission.loc[:, "predict"] = submission["predict"].apply(lambda x: int(round(x)))
submission.loc[:, "predict"] = submission["predict"].astype(int)

# ===== 驗證格式是否正確 =====
unique_vals = submission["predict"].unique()
if not set(unique_vals).issubset({0, 1}):
    raise ValueError(f"❌ 預測結果格式錯誤！detect={unique_vals}")

# ====== 輸出完整預測結果 ======
submission.to_csv(output_full, index=False)
print(f"💾 已輸出完整結果：{output_full}")

# ====== 建立比賽提交版本 ======
submission[["alert_key", "predict"]].to_csv(submit_file, index=False)
print(f"🏁 已輸出比賽用結果：{submit_file}")

# ====== 結果檢查 ======
print("✅ 檢查結果：predict 欄位唯一值 =", submission['predict'].unique())
print("✅ 全部完成，可直接上傳 AI CUP！")
