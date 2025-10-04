# ==========================================
# train_model_LGBM_v3.py (for LightGBM v4.x)
# 功能：自動偵測 GPU，訓練警示帳戶模型
# ==========================================

import pandas as pd
import lightgbm as lgb
import joblib
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, confusion_matrix
)

# ======================
# 路徑設定
# ======================
train_file = "train_data_v3/train_data.csv"
output_dir = "train_data_v3"
os.makedirs(output_dir, exist_ok=True)

model_file = os.path.join(output_dir, "lightgbm_model.pkl")
feature_importance_file = os.path.join(output_dir, "feature_importance.csv")

# ======================
# 1️⃣ 讀取資料
# ======================
print(f"📂 讀取資料：{train_file}")
df = pd.read_csv(train_file)

y = df["alert_flag"]
X = df.drop(columns=["acct_id", "alert_flag"])

# ======================
# 2️⃣ 移除非數值欄位
# ======================
non_numeric_cols = [c for c in X.columns if X[c].dtype not in ["int64", "float64", "bool"]]
if non_numeric_cols:
    print(f"⚠️ 忽略非數值欄位：{non_numeric_cols}")
X = X.select_dtypes(include=["number", "bool"])

print(f"✅ 最終特徵數量：{X.shape[1]}")

# ======================
# 3️⃣ 分割訓練 / 驗證集
# ======================
if y.value_counts().min() < 2:
    print("⚠️ 樣本太少或單一類別，停用 stratify（改為隨機切分）")
    stratify_opt = None
else:
    stratify_opt = y

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=stratify_opt, random_state=42
)

# ======================
# 4️⃣ 嘗試使用 GPU
# ======================
device_type = "cpu"
try:
    test_params = {"device": "gpu"}
    dtest = lgb.Dataset([[0, 1], [1, 0]], label=[0, 1])
    lgb.train(test_params, dtest, num_boost_round=1)
    device_type = "gpu"
    print("⚡ GPU 可用，將使用 GPU 加速訓練")
except Exception:
    print("💡 未偵測到 GPU，改用 CPU 模式")

# ======================
# 5️⃣ LightGBM 參數設定
# ======================
params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbosity": -1,
    "device": device_type
}

# ======================
# 6️⃣ 模型訓練
# ======================
print(f"🚀 開始訓練 LightGBM 模型 ({device_type.upper()}) ...")
start = time.time()

train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

callbacks = [
    lgb.early_stopping(stopping_rounds=100),
    lgb.log_evaluation(period=200)
]

model = lgb.train(
    params=params,
    train_set=train_data,
    num_boost_round=2000,
    valid_sets=[train_data, val_data],
    valid_names=["train", "valid"],
    callbacks=callbacks
)

end = time.time()
print(f"⏱️ 訓練完成，耗時 {end - start:.2f} 秒")

# ======================
# 7️⃣ 模型評估
# ======================
print("📊 模型評估中 ...")
y_pred_prob = model.predict(X_val, num_iteration=model.best_iteration)
thresholds = [i / 100 for i in range(30, 71, 5)]

best_f1 = 0
best_th = 0.5
for th in thresholds:
    y_pred = (y_pred_prob >= th).astype(int)
    f1 = f1_score(y_val, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_th = th

y_pred = (y_pred_prob >= best_th).astype(int)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
acc = accuracy_score(y_val, y_pred)
cm = confusion_matrix(y_val, y_pred)

print("\n✅ 評估結果：")
print(f"Best Threshold : {best_th:.3f}")
print(f"F1 Score       : {best_f1:.4f}")
print(f"Precision      : {precision:.4f}")
print(f"Recall         : {recall:.4f}")
print(f"Accuracy       : {acc:.4f}")
print("Confusion Matrix:")
print(cm)

# ======================
# 8️⃣ 儲存模型與特徵重要度
# ======================
joblib.dump(model, model_file)
print(f"\n💾 模型已儲存至：{model_file}")

feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importance()
}).sort_values(by="importance", ascending=False)
feature_importance.to_csv(feature_importance_file, index=False)
print(f"📊 特徵重要度已輸出至：{feature_importance_file}")
