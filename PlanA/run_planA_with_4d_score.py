import json
from pathlib import Path

import numpy as np
import pandas as pd

import config
import data_loader
import fusion_strategy
import signal_builder
import evaluator


FOUR_DIM_PATH = config.BASE_DIR / "four_dimension_risk_score_old.csv"
OUT_DIR = config.OUTPUT_ROOT_DIR / "output_4d_max"


def main():
    print("===== 读取四维风险分数 =====")

    df = pd.read_csv(
        FOUR_DIM_PATH,
        dtype={"stock_code": str},
        low_memory=False,
    )

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)

    risk_cols = [
        "financial_risk",
        "normative_risk",
        "illegal_risk",
        "other_risk",
    ]

    for col in risk_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df = df.dropna(subset=["date", "stock_code"])

    print("原始四维数据行数:", len(df))

    # ====================================================
    # 用 max 作为最终风险分
    # ====================================================
    print("===== 四维 max 融合 =====")

    df["risk_score"] = df[risk_cols].max(axis=1)

    print("综合分非零统计：")

    nonzero = df.loc[df["risk_score"] > 0, "risk_score"]

    print(
        nonzero.describe(
            percentiles=[0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
        )
    )

    print("===== 构建 index 和标签，不读取公告 =====")

    start_date = pd.Timestamp(config.OUTPUT_START_DATE).normalize()
    end_date = pd.Timestamp(config.OUTPUT_END_DATE).normalize()

    calendar = pd.date_range(start_date, end_date, freq="D")

    universe = sorted(df["stock_code"].dropna().unique())

    full_index = pd.MultiIndex.from_product(
        [universe, calendar],
        names=["stock_code", "date"],
    )

    data = pd.DataFrame(index=full_index)

    print("===== 同股票同日期聚合综合分 =====")

    daily_score = (
        df.groupby(["stock_code", "date"])["risk_score"]
        .max()
    )

    raw_score = (
        daily_score
        .reindex(full_index)
        .fillna(0)
        .astype(np.float32)
    )

    print("日度 raw_score 统计：")

    print(
        raw_score.describe(
            percentiles=[0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
        )
    )

    print("===== 读取并构建 ST 标签 =====")

    st_df = data_loader.load_raw_ST()
    st_df = data_loader.clean_ST(st_df)

    y_status = pd.Series(0, index=full_index, dtype=np.int8)

    universe_set = set(universe)

    for _, row in st_df.iterrows():

        code = str(row["stock_code"]).zfill(6)

        if code not in universe_set:
            continue

        entry = pd.to_datetime(row["entry_date"], errors="coerce")
        remove = pd.to_datetime(row["remove_date"], errors="coerce")

        if pd.isna(entry):
            continue

        entry = entry.normalize()

        if pd.isna(remove):
            remove = end_date
        else:
            remove = remove.normalize()

        s = max(entry, start_date)
        e = min(remove, end_date)

        if s > e:
            continue

        mask_dates = (calendar >= s) & (calendar <= e)

        if mask_dates.any():
            y_status.loc[(code, calendar[mask_dates])] = 1

    data["y_status"] = y_status

    print("ST 状态样本数:", int(data["y_status"].sum()))

    # ====================================================
    # PlanA 双阈值 + decay
    # ====================================================
    print("===== 调用 PlanA 双阈值 decay 机制 =====")

    output_mask = np.ones(len(full_index), dtype=bool)

    score_all = fusion_strategy.apply_decay_memory_scores(
        raw_score_s=raw_score,
        output_mask=output_mask,
        decay_per_day=config.FUSION_DECAY_PER_DAY,
        entry_threshold=config.BLACKLIST_ENTRY_THRESHOLD,
        retain_threshold=config.BLACKLIST_RETAIN_THRESHOLD,
    )

    print("decay 后 score_all 统计：")

    print(
        score_all.describe(
            percentiles=[0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
        )
    )

    print("===== 生成黑名单 JSON =====")

    threshold = config.BLACKLIST_RETAIN_THRESHOLD

    out_blacklist = signal_builder.build_blacklist_json_threshold(
        data=data,
        score=score_all,
        threshold=threshold,
        topk_cap=None,
    )

    print("===== 评估指标 =====")

    metrics, daily_df = evaluator.evaluate_screenshot_metrics_threshold(
        data=data,
        score=score_all,
        threshold=threshold,
        topk_cap=None,
    )

    print("metrics:")
    print(metrics)

    print("===== 保存结果 =====")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    submit_path = OUT_DIR / "PlanA_4d_max_decay.json"

    metrics_path = OUT_DIR / "PlanA_4d_max_decay_metrics.json"

    daily_path = OUT_DIR / "PlanA_4d_max_decay_daily.csv"

    with open(submit_path, "w", encoding="utf-8") as f:
        json.dump(out_blacklist, f, ensure_ascii=False, indent=2)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    daily_df.to_csv(daily_path, encoding="utf-8-sig")

    print("submit:", submit_path)
    print("metrics:", metrics_path)
    print("daily:", daily_path)

    print("===== 完成 =====")


if __name__ == "__main__":
    main()
