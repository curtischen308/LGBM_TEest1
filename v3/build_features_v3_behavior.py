# ===============================================
# build_features_v3_behavior.py
# 功能：
#   1. 從 acct_transaction.csv 建立帳號級特徵
#   2. 新增行為／時間／金額／通道等動態特徵
#   3. 修正日期格式警告 + 處理 UNK 類別
#   4. 輸出乾淨可用於 LightGBM 的數值特徵
# Step 2️⃣ 製作 1:1 訓練集
#python v3/prepare_trainset_balanced_v3.py

# Step 3️⃣ 訓練模型
#python v3/train_model_LGBM_v3.py

# Step 4️⃣ 預測比賽帳號
#python v3/predict_accounts_v3.py

# ===============================================

import pandas as pd
import numpy as np
import os


def build_features_v3(input_file: str, output_dir: str = "feature_data_v3"):
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "account_features.csv")

    print(f"📂 讀取交易資料：{input_file}")
    df = pd.read_csv(input_file)

    # ========== 類別欄位清理 ==========
    print("🧹 清理類別欄位（UNK / NaN 轉換） ...")
    cat_cols = ["from_acct_type", "to_acct_type", "currency_type", "channel_type"]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].fillna("UNK").astype(str)
            df[c] = df[c].replace("UNK", "-1")  # 保留未知類別為 -1

    # ========== 新增 is_unk 標記 ==========
    df["is_channel_unk"] = (df["channel_type"] == "-1").astype(int)
    df["is_toacct_unk"] = (df["to_acct_type"] == "-1").astype(int)

    # ========== 日期時間處理（修正警告） ==========
    print("🕒 處理日期與時間欄位 ...")
    base_date = pd.Timestamp("2020-01-01")
    df["txn_datetime"] = base_date + pd.to_timedelta(df["txn_date"].astype(int), unit="D")
    df["txn_datetime"] += pd.to_timedelta(df["txn_time"])

    # ========== 基本統計特徵 ==========
    print("📊 建立基本交易統計 ...")
    g = df.groupby("from_acct")
    from_feat = g["txn_amt"].agg(
        txn_count="count",
        total_amt="sum",
        avg_amt="mean",
        max_amt="max",
        min_amt="min",
        std_amt="std"
    ).reset_index()

    # ========== 時間行為特徵 ==========
    print("⏰ 加入時間行為特徵 ...")
    txn_time_feat = g["txn_datetime"].agg(first_txn="min", last_txn="max")
    txn_time_feat["active_days"] = (txn_time_feat["last_txn"] - txn_time_feat["first_txn"]).dt.days + 1
    txn_time_feat = txn_time_feat.reset_index()
    from_feat = from_feat.merge(txn_time_feat, on="from_acct", how="left")
    from_feat["txn_per_day"] = from_feat["txn_count"] / from_feat["active_days"].replace(0, 1)

    # 夜間交易比例
    df["hour"] = pd.to_datetime(df["txn_time"], format="%H:%M:%S", errors="coerce").dt.hour
    df["is_night"] = df["hour"].apply(lambda x: 1 if x >= 22 or x < 6 else 0)
    night_ratio = g["is_night"].mean().reset_index().rename(columns={"is_night": "night_txn_ratio"})
    from_feat = from_feat.merge(night_ratio, on="from_acct", how="left")

    # ========== 自轉與重複對象特徵 ==========
    print("🔁 分析自轉與重複對象特徵 ...")
    df["is_self_txn"] = (df["from_acct"] == df["to_acct"]).astype(int)
    self_ratio = g["is_self_txn"].mean().reset_index().rename(columns={"is_self_txn": "self_txn_ratio"})
    from_feat = from_feat.merge(self_ratio, on="from_acct", how="left")

    # 重複對手比例
    to_count = g["to_acct"].count().reset_index(name="to_txn_total")
    unique_to = g["to_acct"].nunique().reset_index(name="to_acct_nunique")
    repeat_ratio = to_count.merge(unique_to, on="from_acct")
    repeat_ratio["repeat_partner_ratio"] = 1 - (repeat_ratio["to_acct_nunique"] / repeat_ratio["to_txn_total"])
    from_feat = from_feat.merge(repeat_ratio, on="from_acct", how="left")

    # ========== 金額異常度特徵 ==========
    print("💰 加入金額異常度特徵 ...")
    from_feat["max_to_avg"] = from_feat["max_amt"] / (from_feat["avg_amt"] + 1e-5)
    from_feat["amount_cv"] = from_feat["std_amt"] / (from_feat["avg_amt"] + 1e-5)

    # ========== 通道多樣性 ==========
    print("🌐 通道多樣性特徵 ...")
    channel_div = g["channel_type"].nunique().reset_index().rename(columns={"channel_type": "channel_nunique"})
    from_feat = from_feat.merge(channel_div, on="from_acct", how="left")

    # ========== 對手帳戶多樣性 ==========
    print("🔗 對手帳戶多樣性 ...")
    from_feat["partner_diversity"] = from_feat["to_acct_nunique"] / from_feat["txn_count"].replace(0, 1)

    # ========== UNK 類別比例特徵 ==========
    print("❓ 加入未知類別比例特徵 ...")
    unk_feat = df.groupby("from_acct")[["is_channel_unk", "is_toacct_unk"]].mean().reset_index()
    from_feat = from_feat.merge(unk_feat, on="from_acct", how="left")

    # ========== 輸出 ==========
    from_feat.rename(columns={"from_acct": "acct_id"}, inplace=True)
    from_feat.fillna(0, inplace=True)
    from_feat.to_csv(output_file, index=False)

    print(f"✅ 完成，共 {len(from_feat)} 筆帳號特徵")
    print(f"💾 已輸出至：{output_file}")


if __name__ == "__main__":
    input_path = "dataset/acct_transaction.csv"
    build_features_v3(input_path)
