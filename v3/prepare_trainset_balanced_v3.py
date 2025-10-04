# ==========================================
# prepare_trainset_balanced_v3.py
# 功能：從特徵表中取出警示帳戶，並欠採樣正常帳戶成 1:1 訓練集
# ==========================================

import pandas as pd
import os

# ======================
# 檔案路徑
# ======================
feature_file = "feature_data_v3/account_features.csv"
alert_file = "dataset/acct_alert.csv"
output_dir = "train_data_v3"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "train_data.csv")

# ======================
# 讀取資料
# ======================
print(f"📂 讀取特徵檔案：{feature_file}")
features = pd.read_csv(feature_file)

print(f"📂 讀取警示帳戶：{alert_file}")
alerts = pd.read_csv(alert_file)

# ======================
# 欄位比對
# ======================
if "acct" in alerts.columns:
    alert_col = "acct"
elif "alert_key" in alerts.columns:
    alert_col = "alert_key"
else:
    raise ValueError("❌ acct_alert.csv 缺少帳號欄位（acct 或 alert_key）")

alert_accounts = set(alerts[alert_col])
features["alert_flag"] = features["acct_id"].isin(alert_accounts).astype(int)

alert_df = features[features["alert_flag"] == 1]
normal_df = features[features["alert_flag"] == 0]

print(f"🚨 警示帳戶數：{len(alert_df)}")
print(f"🟢 正常帳戶數：{len(normal_df)}")

# ======================
# 欠採樣 (1:1)
# ======================
normal_sample = normal_df.sample(n=len(alert_df), random_state=42)
train_data = pd.concat([alert_df, normal_sample]).sample(frac=1, random_state=42)

print(f"✅ 欠採樣後訓練集共 {len(train_data)} 筆資料 (1:1 比例)")
train_data.to_csv(output_file, index=False)
print(f"💾 已輸出至：{output_file}")
