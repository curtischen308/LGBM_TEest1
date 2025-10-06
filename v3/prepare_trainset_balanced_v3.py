# ===============================================
# prepare_trainset_balanced_v3_dual.py
# 功能：
#   ✅ 從雙向帳號特徵表建立訓練集
#   ✅ 警示帳號 = acct_alert.csv
#   ✅ 欠採樣正常帳號形成 1:1 比例
# ===============================================

import pandas as pd
import os

# ======================
# 檔案路徑設定
# ======================
feature_file = "feature_data_v3/account_features.csv"
alert_file = "dataset/acct_alert.csv"
output_dir = "train_data_v3"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "train_data.csv")

# ======================
# 1️⃣ 讀取資料
# ======================
print(f"📂 讀取特徵檔案：{feature_file}")
features = pd.read_csv(feature_file)

print(f"📂 讀取警示帳戶清單：{alert_file}")
alerts = pd.read_csv(alert_file)

# ======================
# 2️⃣ 尋找帳號欄位
# ======================
if "acct" in alerts.columns:
    alert_col = "acct"
elif "alert_key" in alerts.columns:
    alert_col = "alert_key"
else:
    raise ValueError("❌ acct_alert.csv 缺少帳號欄位（acct 或 alert_key）")

alerts[alert_col] = alerts[alert_col].astype(str).str.strip()
features["acct_id"] = features["acct_id"].astype(str).str.strip()

# ======================
# 3️⃣ 標註警示標籤
# ======================
features["alert_flag"] = features["acct_id"].isin(alerts[alert_col]).astype(int)

alert_df = features[features["alert_flag"] == 1]
normal_df = features[features["alert_flag"] == 0]

print(f"🚨 警示帳戶數：{len(alert_df)}")
print(f"🟢 正常帳戶數：{len(normal_df)}")

# ======================
# 4️⃣ 欠採樣 (1:1 比例)
# ======================
if len(alert_df) == 0:
    raise ValueError("❌ 警示帳號為 0，請確認 acct_alert.csv 是否正確。")

normal_sample = (
    normal_df.sample(n=len(alert_df), random_state=42)
    if len(normal_df) >= len(alert_df)
    else normal_df
)

train_data = pd.concat([alert_df, normal_sample]).sample(frac=1, random_state=42)

print(f"✅ 欠採樣後訓練集共 {len(train_data)} 筆資料 (1:1 比例)")
print(f"📊 其中警示帳戶：{train_data['alert_flag'].sum()} 筆")

# ======================
# 5️⃣ 輸出
# ======================
train_data.to_csv(output_file, index=False)
print(f"💾 已輸出至：{output_file}")
