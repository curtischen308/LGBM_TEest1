# ==========================================================
# predict_accounts_v3_dual.py
# 功能：
#   ✅ 自動載入最佳閾值 best_threshold.json
#   ✅ 自動補缺帳號特徵（以平均值填補）
#   ✅ 自動偵測 GPU / CPU
#   ✅ 確保 acct 對齊不會全為 0
# ==========================================================

import pandas as pd
import numpy as np
import joblib
import os
import json

# ==========================
# 1️⃣ 路徑設定
# ==========================
feature_file = "feature_data_v3/account_features.csv"
predict_file = "dataset/acct_predict.csv"
model_file = "train_data_v3/lightgbm_model.pkl"
threshold_file = "train_data_v3/best_threshold.json"
output_dir = "results_v3"
os.makedirs(output_dir, exist_ok=True)

output_full = os.path.join(output_dir, "predict_full_final.csv")
submit_file = os.path.join(output_dir, "predict_for_submit_final.csv")

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
# 3️⃣ 統一帳號欄位名稱
# ==========================
acct_col = None
for col in predict_accts.columns:
    if col.strip().lower() in ["acct", "acct_id"]:
        acct_col = col
        break
if acct_col is None:
    raise ValueError("❌ acct_predict.csv 中沒有找到帳號欄位 (acct 或 acct_id)")

predict_accts.rename(columns={acct_col: "acct_id"}, inplace=True)
features["acct_id"] = features["acct_id"].astype(str).str.strip()
predict_accts["acct_id"] = predict_accts["acct_id"].astype(str).str.strip()

# ==========================
# 4️⃣ 轉換時間欄位
# ==========================
for col in ["first_txn", "last_txn"]:
    if col in features.columns:
        features[col] = pd.to_datetime(features[col], errors="coerce")
        features[col] = (features[col] - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
        print(f"🕒 已將欄位 {col} 轉換為 UNIX 秒數格式")

# ==========================
# 5️⃣ 對齊特徵欄位
# ==========================
train_features = model.feature_name()
extra_cols = [c for c in features.columns if c not in train_features]
if extra_cols:
    print(f"⚠️ 多出欄位: {extra_cols}")

# 自動補缺失欄位
for c in train_features:
    if c not in features.columns:
        features[c] = 0

# ==========================
# 6️⃣ 比對帳號覆蓋率
# ==========================
matched = features["acct_id"].isin(predict_accts["acct_id"])
unmatched_count = len(predict_accts) - matched.sum()
if unmatched_count > 0:
    print(f"⚠️ 有 {unmatched_count} 筆帳號沒有特徵，將自動補平均值特徵")

# 複製平均特徵作為補值樣板
mean_row = features[train_features].mean()
missing_accts = predict_accts.loc[~predict_accts["acct_id"].isin(features["acct_id"]), "acct_id"]
missing_features = pd.DataFrame([mean_row] * len(missing_accts))
missing_features.insert(0, "acct_id", missing_accts.values)

# 合併補足後完整特徵表
features_fixed = pd.concat([features, missing_features], ignore_index=True)

# 選出預測名單
predict_features = features_fixed[features_fixed["acct_id"].isin(predict_accts["acct_id"])].copy()
X_pred = predict_features.reindex(columns=train_features, fill_value=0)

print(f"🔹 使用特徵數量: {X_pred.shape[1]}，預測筆數: {len(X_pred)}")

# ==========================
# 7️⃣ 載入最佳閾值
# ==========================
threshold = 0.5
if os.path.exists(threshold_file):
    with open(threshold_file, "r") as f:
        data = json.load(f)
        threshold = data.get("best_threshold", 0.5)
    print(f"🧠 自動載入訓練最佳閾值: {threshold:.3f}")
else:
    print(f"⚠️ 未找到 {threshold_file}，使用預設 0.5")

# ==========================
# 8️⃣ 預測階段
# ==========================
device_info = "GPU" if "gpu" in str(model.params.get("device", "")).lower() else "CPU"
print(f"🚀 使用 {device_info} 進行預測中 ...")

y_pred_prob = model.predict(X_pred, num_iteration=getattr(model, "best_iteration", None))
y_pred_prob = np.array(y_pred_prob).flatten()

predict_features["probability"] = y_pred_prob
predict_features["label"] = (y_pred_prob >= threshold).astype(int)

print(f"✅ 已完成模型預測，共 {len(predict_features)} 筆。")

# ==========================
# 🔧 合併防呆：統一帳號格式
# ==========================
predict_accts["acct_id"] = (
    predict_accts["acct_id"].astype(str).str.strip().str.lower()
)
predict_features["acct_id"] = (
    predict_features["acct_id"].astype(str).str.strip().str.lower()
)

# 比對交集與差集
common = set(predict_accts["acct_id"]) & set(predict_features["acct_id"])
missing = set(predict_accts["acct_id"]) - common
print(f"📊 匹配帳號數：{len(common)} / {len(predict_accts)}")
if len(missing) > 0:
    print(f"⚠️ 有 {len(missing)} 筆帳號仍無法對上特徵（將自動補0）")
    print("🔍 前5筆未匹配帳號：", list(missing)[:5])

# ==========================
# 🔗 合併回 acct_predict 名單 ...
# ==========================
print("🔗 合併回 acct_predict 名單 ...")

submission = pd.merge(
    predict_accts,
    predict_features[["acct_id", "label", "probability"]],
    on="acct_id",
    how="left"
)

# 🧩 修正重複欄位（label_x, label_y）
if "label_x" in submission.columns and "label_y" in submission.columns:
    print("⚙️ 偵測到 label_x / label_y，進行合併 ...")
    submission["label"] = submission["label_y"].combine_first(submission["label_x"])
    submission.drop(columns=["label_x", "label_y"], inplace=True)

# 🧩 若仍無 label 欄位 → 補上 0
if "label" not in submission.columns:
    print("❌ 警告：合併後完全沒有 label 欄位，將補上全 0。")
    submission["label"] = 0

# 填補缺值
submission["label"] = submission["label"].fillna(0).astype(int)
submission["probability"] = submission["probability"].fillna(0)

# ==========================
# 🔟 輸出結果
# ==========================
submission.rename(columns={"acct_id": "acct"}, inplace=True)
submission["acct"] = submission["acct"].astype(str)
submission["label"] = submission["label"].astype(int)

submission.to_csv(output_full, index=False)
submission[["acct", "label"]].to_csv(submit_file, index=False)

# ==========================
# 📊 結果摘要
# ==========================
print(f"💾 已輸出完整結果：{output_full}")
print(f"🏁 已輸出比賽用結果（acct,label 格式）：{submit_file}")
print("✅ label 唯一值 =", submission["label"].unique())
print("✅ acct 數量 =", len(submission))
print("📊 label 分布：")
print(submission["label"].value_counts(normalize=True))

print("✅ 完成，可直接上傳 AI CUP！")
