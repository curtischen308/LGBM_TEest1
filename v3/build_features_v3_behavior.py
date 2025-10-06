# ===============================================
# build_features_v3_behavior_dual.py
# 功能：
#   ✅ 同時考慮匯款帳戶與收款帳戶，擴大覆蓋率
#   ✅ 修正 acct_predict 對不到特徵的問題
#   ✅ 加入金額、時間、行為、通道、UNK 特徵
# ===============================================

import pandas as pd
import numpy as np
import os


def build_features_v3_dual(input_file: str, output_dir: str = "feature_data_v3"):
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "account_features.csv")

    print(f"📂 讀取交易資料：{input_file}")
    df = pd.read_csv(input_file)

    # ========== 1️⃣ 基本清理 ==========
    print("🧹 清理類別欄位（UNK / NaN 轉換） ...")
    cat_cols = ["from_acct_type", "to_acct_type", "currency_type", "channel_type"]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].fillna("UNK").astype(str)
            df[c] = df[c].replace("UNK", "-1")

    df["is_channel_unk"] = (df["channel_type"] == "-1").astype(int)
    df["is_toacct_unk"] = (df["to_acct_type"] == "-1").astype(int)

    # ========== 2️⃣ 雙向帳號合併 ==========
    print("🔁 建立雙向帳號資料 ...")
    df_from = df.rename(columns={"from_acct": "acct_id"})
    df_to = df.rename(columns={"to_acct": "acct_id"})
    df_all = pd.concat([df_from, df_to], ignore_index=True)
    df_all["txn_amt"] = pd.to_numeric(df_all["txn_amt"], errors="coerce").fillna(0)

    # ========== 3️⃣ 日期時間處理 ==========
    print("🕒 處理日期與時間欄位 ...")
    base_date = pd.Timestamp("2020-01-01")
    df_all["txn_datetime"] = base_date + pd.to_timedelta(df_all["txn_date"].astype(int), unit="D")
    df_all["txn_datetime"] += pd.to_timedelta(df_all["txn_time"], errors="coerce")

    # ========== 4️⃣ 基本統計特徵 ==========
    print("📊 建立基本交易統計 ...")
    g = df_all.groupby("acct_id")
    feat = g["txn_amt"].agg(
        txn_count="count",
        total_amt="sum",
        avg_amt="mean",
        max_amt="max",
        min_amt="min",
        std_amt="std"
    ).reset_index()

    # ========== 5️⃣ 時間行為特徵 ==========
    print("⏰ 加入時間行為特徵 ...")
    txn_time_feat = g["txn_datetime"].agg(first_txn="min", last_txn="max")
    txn_time_feat["active_days"] = (txn_time_feat["last_txn"] - txn_time_feat["first_txn"]).dt.days + 1
    txn_time_feat = txn_time_feat.reset_index()
    feat = feat.merge(txn_time_feat, on="acct_id", how="left")
    feat["txn_per_day"] = feat["txn_count"] / feat["active_days"].replace(0, 1)

    # 夜間比例
    df_all["hour"] = pd.to_datetime(df_all["txn_time"], format="%H:%M:%S", errors="coerce").dt.hour
    df_all["is_night"] = df_all["hour"].apply(lambda x: 1 if x >= 22 or x < 6 else 0)
    night_ratio = g["is_night"].mean().reset_index().rename(columns={"is_night": "night_txn_ratio"})
    feat = feat.merge(night_ratio, on="acct_id", how="left")

    # ========== 6️⃣ 自轉與重複對象特徵 ==========
    print("🔄 自轉與對手多樣性特徵 ...")
    df_all["is_self_txn"] = (df_all["from_acct"] == df_all["to_acct"]).astype(int)
    self_ratio = df_all.groupby("acct_id")["is_self_txn"].mean().reset_index().rename(
        columns={"is_self_txn": "self_txn_ratio"}
    )
    feat = feat.merge(self_ratio, on="acct_id", how="left")

    # ========== 7️⃣ 金額異常度 ==========
    print("💰 加入金額異常度特徵 ...")
    feat["max_to_avg"] = feat["max_amt"] / (feat["avg_amt"] + 1e-5)
    feat["amount_cv"] = feat["std_amt"] / (feat["avg_amt"] + 1e-5)

    # ========== 8️⃣ 通道多樣性 ==========
    print("🌐 通道多樣性特徵 ...")
    channel_div = g["channel_type"].nunique().reset_index().rename(columns={"channel_type": "channel_nunique"})
    feat = feat.merge(channel_div, on="acct_id", how="left")

    # ========== 9️⃣ UNK 比例 ==========
    print("❓ 加入未知類別比例特徵 ...")
    unk_feat = g[["is_channel_unk", "is_toacct_unk"]].mean().reset_index()
    feat = feat.merge(unk_feat, on="acct_id", how="left")

    # ========== 🔟 輸出 ==========
    feat.fillna(0, inplace=True)
    feat.to_csv(output_file, index=False)
    print(f"✅ 完成，共 {len(feat)} 筆帳號特徵")
    print(f"💾 已輸出至：{output_file}")


if __name__ == "__main__":
    input_path = "dataset/acct_transaction.csv"
    build_features_v3_dual(input_path)
