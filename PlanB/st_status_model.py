import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else range(0)


RISK_COLUMNS = ["financial_risk", "normative_risk", "illegal_risk", "other_risk"]

BASE_DIR = Path(__file__).resolve().parent

# Explicit input CSV paths.
RISK_SCORE_CSV = BASE_DIR / "four_dimension_risk_score.csv"
ST_URGENCY_CSV = BASE_DIR / "ST_history_label.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "st_status_model"


def _normalize_code(value):
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if "." in text:
        text = text.split(".", 1)[0]
    if text.endswith(".0"):
        text = text[:-2]
    if text.isdigit():
        return text.zfill(6)
    return text


def _read_risk_scores(path, start_date, end_date):
    df = pd.read_csv(
        path,
        dtype={"stock_code": "string"},
        parse_dates=["date"],
        low_memory=False,
    )
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["stock_code", "date"])
    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()
    df["stock_code"] = df["stock_code"].map(_normalize_code)

    for col in RISK_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(0, 100).astype("float32")

    df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)
    return df


def _read_st_label_targets(path, start_date, end_date):
    wide = pd.read_csv(path, low_memory=False)
    date_col = wide.columns[0]
    wide[date_col] = pd.to_datetime(wide[date_col], errors="coerce").dt.normalize()
    wide = wide[(wide[date_col] >= start_date) & (wide[date_col] <= end_date)].copy()

    long_df = wide.melt(id_vars=date_col, var_name="stock_code", value_name="raw_st_label")
    long_df = long_df.rename(columns={date_col: "date"})
    long_df["stock_code"] = long_df["stock_code"].map(_normalize_code)
    long_df["label_in_file"] = np.int8(1)
    long_df["y_label_empty"] = long_df["raw_st_label"].isna().astype(np.int8)
    long_df["y_label_value"] = (
        pd.to_numeric(long_df["raw_st_label"], errors="coerce")
        .clip(0, 1)
        .astype("float32")
    )
    long_df = long_df[["date", "stock_code", "label_in_file", "y_label_empty", "y_label_value"]]
    return long_df


def _empty_label_frame():
    return pd.DataFrame(
        {
            "date": pd.Series(dtype="datetime64[ns]"),
            "stock_code": pd.Series(dtype="object"),
            "label_in_file": pd.Series(dtype="int8"),
            "y_label_empty": pd.Series(dtype="int8"),
            "y_label_value": pd.Series(dtype="float32"),
        }
    )


def _add_targets(df, labels):
    df = df.merge(labels, on=["stock_code", "date"], how="left")
    df["label_in_file"] = df["label_in_file"].fillna(0).astype(np.int8)
    df["y_label_empty"] = df["y_label_empty"].fillna(0).astype(np.int8)
    df["y_label_value"] = df["y_label_value"].where(df["y_label_empty"] == 0)
    df.loc[(df["label_in_file"] == 0) & (df["y_label_empty"] == 0), "y_label_value"] = 0
    df["y_label_numeric_available"] = df["y_label_value"].notna().astype(np.int8)
    return df


def _add_features(df):
    df["month"] = df["date"].dt.month.astype(np.int8)
    df["quarter"] = df["date"].dt.quarter.astype(np.int8)
    df["dayofyear"] = df["date"].dt.dayofyear.astype(np.int16)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12).astype("float32")
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12).astype("float32")

    df["risk_max"] = df[RISK_COLUMNS].max(axis=1).astype("float32")
    df["risk_nonzero_count"] = (df[RISK_COLUMNS] > 0).sum(axis=1).astype(np.int8)

    month_key = df["date"].dt.to_period("M")
    for col in RISK_COLUMNS:
        df[f"{col}_month_mean"] = (
            df.groupby(["stock_code", month_key], sort=False)[col].transform("mean").astype("float32")
        )

    code_cat = pd.Categorical(df["stock_code"])
    df["stock_code_id"] = code_cat.codes.astype(np.int16)
    return df


def _feature_columns():
    return [
        *RISK_COLUMNS,
        "risk_max",
        "risk_nonzero_count",
        *[f"{col}_month_mean" for col in RISK_COLUMNS],
        "month",
        "quarter",
        "dayofyear",
        "month_sin",
        "month_cos",
        "stock_code_id",
    ]


def _sample_regression_training_frame(df, target_col, zero_ratio, random_state):
    positive = df[df[target_col] > 0]
    zero = df[df[target_col] == 0]
    n_zero = min(len(zero), max(int(len(positive) * zero_ratio), 1))
    if n_zero < len(zero):
        zero = zero.sample(n=n_zero, random_state=random_state)
    train_df = pd.concat([positive, zero], axis=0).sample(frac=1.0, random_state=random_state)
    return train_df


def _make_regressor(random_state):
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.015708,       # 新超参数
        max_iter=1,
        max_leaf_nodes=21,            # 新超参数
        l2_regularization=0.000196,   # 新超参数
        random_state=random_state,
        warm_start=True,
        early_stopping=False,
    )


def _fit_model_with_tqdm(model, x, y, max_iter=96, desc="训练模型"):
    for n_iter in tqdm(range(1, max_iter + 1), desc=desc, unit="iter"):
        model.set_params(max_iter=n_iter)
        model.fit(x, y)
    return model


def _train_value_model(df, features, validation_start, zero_ratio, random_state, max_iter):
    target_col = "y_label_value"
    work = df[df["y_label_numeric_available"] == 1].copy()
    validation_start = pd.Timestamp(validation_start).normalize()
    train_part = work[work["date"] < validation_start]
    valid_part = work[work["date"] >= validation_start]
    if train_part[target_col].nunique() < 2 or valid_part[target_col].nunique() < 2:
        train_part = work.sample(frac=0.8, random_state=random_state)
        valid_part = work.drop(train_part.index)

    sampled_train = _sample_regression_training_frame(train_part, target_col, zero_ratio, random_state)
    final_sample = _sample_regression_training_frame(work, target_col, zero_ratio, random_state)

    # 用于验证集评估的模型
    val_model = _fit_model_with_tqdm(
        _make_regressor(random_state),
        sampled_train[features],
        sampled_train[target_col],
        max_iter=max_iter,
        desc="训练验证回归模型",
    )
    valid_pred = np.clip(val_model.predict(valid_part[features]), 0, 1)

    # 最终模型
    final_model = _fit_model_with_tqdm(
        _make_regressor(random_state),
        final_sample[features],
        final_sample[target_col],
        max_iter=max_iter,
        desc="训练最终回归模型",
    )
    rmse = float(np.sqrt(mean_squared_error(valid_part[target_col], valid_pred)))
    metrics = {
        "value_validation_mae": float(mean_absolute_error(valid_part[target_col], valid_pred)),
        "value_validation_rmse": rmse,
        "value_training_rows_sampled": int(len(final_sample)),
        "value_positive_rows_total": int((work[target_col] > 0).sum()),
        "value_zero_rows_total": int((work[target_col] == 0).sum()),
        "value_numeric_rows_total": int(len(work)),
    }
    return final_model, metrics


def _daily_json(df, prediction_col="predicted_st"):
    pred = df[df[prediction_col] == 1].copy()
    pred = pred.sort_values(["date", "stock_code"])
    daily = {}
    all_dates = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    grouped = pred.groupby("date")["stock_code"].apply(list).to_dict()
    for date in all_dates:
        daily[str(date.date())] = grouped.get(date, [])
    return daily


def _interval_json(df, prediction_col="predicted_st"):
    intervals = []
    for code, g in df.sort_values(["stock_code", "date"]).groupby("stock_code", sort=False):
        flag = g[prediction_col].to_numpy(np.int8)
        if flag.sum() == 0:
            continue
        dates = g["date"].to_numpy("datetime64[ns]")
        prev = np.r_[0, flag[:-1]]
        nxt = np.r_[flag[1:], 0]
        starts = np.flatnonzero((flag == 1) & (prev == 0))
        ends = np.flatnonzero((flag == 1) & (nxt == 0))
        for start_i, end_i in zip(starts, ends):
            intervals.append(
                {
                    "stock_code": str(code),
                    "entry_date": str(pd.Timestamp(dates[start_i]).date()),
                    "remove_date": str(pd.Timestamp(dates[end_i]).date()),
                }
            )
    return intervals


def _label_json(df, value_col="predicted_label", empty_col="predicted_label_empty"):
    out = {}
    for date, g in df.sort_values(["date", "stock_code"]).groupby("date", sort=False):
        day = {}
        for code, value, is_empty in zip(g["stock_code"], g[value_col], g[empty_col]):
            day[str(code)] = None if int(is_empty) == 1 else round(float(value), 6)
        out[str(pd.Timestamp(date).date())] = day
    return out


def run(args):
    train_start_date = pd.Timestamp(args.train_start_date).normalize()
    train_end_date = pd.Timestamp(args.train_end_date).normalize()
    predict_start_date = pd.Timestamp(args.predict_start_date).normalize()
    predict_end_date = pd.Timestamp(args.predict_end_date).normalize()
    read_start_date = min(train_start_date, predict_start_date)
    read_end_date = max(train_end_date, predict_end_date)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("输入 CSV 路径:")
    print(f"  四维风险评分: {Path(args.risk_score_csv).resolve()}")
    print(f"  ST距离/空值标签: {Path(args.st_label_csv).resolve()}")

    print("[1/6] 读取四维风险评分")
    df = _read_risk_scores(args.risk_score_csv, read_start_date, read_end_date)

    print("[2/6] 合并 ST_history_label.csv 监督目标")
    labels = _read_st_label_targets(args.st_label_csv, train_start_date, train_end_date)
    if labels.empty:
        labels = _empty_label_frame()
    df = _add_targets(df, labels)

    print("[3/6] 构造风险和日期特征")
    df = _add_features(df)
    features = _feature_columns()
    train_df = df[(df["date"] >= train_start_date) & (df["date"] <= train_end_date)].copy()

    print("[4/6] 训练紧急性回归模型")
    value_model, value_metrics = _train_value_model(
        df=train_df,
        features=features,
        validation_start=args.validation_start,
        zero_ratio=args.zero_ratio,
        random_state=args.random_state,
        max_iter=args.max_iter,
    )
    metrics = {
        "validation_start": str(pd.Timestamp(args.validation_start).normalize().date()),
        "max_iter": int(args.max_iter),
        **value_metrics,
    }

    print("[5/6] 生成预测标签与黑名单")
    df["predicted_label_value"] = np.clip(value_model.predict(df[features]), 0, 1).astype("float32")
    # 不再使用分类器，所有样本均视为非空值（空值信息通过已知ST状态处理）
    df["predicted_label_empty"] = np.int8(0)
    df["predicted_label"] = df["predicted_label_value"]
    # predicted_st 直接使用真实标签（历史期有效，预测期填充0）
    df["predicted_st"] = df["y_label_empty"].fillna(0).astype(np.int8)
    # 黑名单：样本非ST状态（y_label_empty != 1，预测期NaN视为0）且预测分超过阈值
    df["predicted_blacklist"] = (
        (df["y_label_empty"].fillna(0) == 0) & (df["predicted_label_value"] >= args.label_threshold)
    ).astype(np.int8)

    predict_label_mask = (df["date"] >= predict_start_date) & (df["date"] <= predict_end_date)
    train_label_mask = (df["date"] >= train_start_date) & (df["date"] <= train_end_date)
    prediction_only_mask = predict_label_mask & ~train_label_mask
    df.loc[prediction_only_mask, ["label_in_file", "y_label_empty", "y_label_value", "y_label_numeric_available"]] = pd.NA

    history_df = df[(df["date"] >= train_start_date) & (df["date"] <= train_end_date)].copy()
    predict_2024_df = df[(df["date"] >= predict_start_date) & (df["date"] <= predict_end_date)].copy()

    print("[6/6] 写出 JSON 和模型文件")
    history_daily_path = output_dir / args.history_daily_json_name
    predict_daily_path = output_dir / args.predict_2024_daily_json_name
    history_interval_path = output_dir / args.history_interval_json_name
    predict_interval_path = output_dir / args.predict_2024_interval_json_name
    blacklist_daily_path = output_dir / args.blacklist_daily_json_name
    predict_label_json_path = output_dir / args.predict_2024_label_json_name
    model_path = output_dir / "st_label_model.joblib"
    metrics_path = output_dir / "st_status_model_metrics.json"
    prediction_csv_path = output_dir / "st_status_predictions.csv"

    with history_daily_path.open("w", encoding="utf-8") as f:
        json.dump(_daily_json(history_df), f, ensure_ascii=False, indent=2)

    with predict_daily_path.open("w", encoding="utf-8") as f:
        json.dump(_daily_json(predict_2024_df), f, ensure_ascii=False, indent=2)

    with history_interval_path.open("w", encoding="utf-8") as f:
        json.dump(_interval_json(history_df), f, ensure_ascii=False, indent=2)

    with predict_interval_path.open("w", encoding="utf-8") as f:
        json.dump(_interval_json(predict_2024_df), f, ensure_ascii=False, indent=2)

    with blacklist_daily_path.open("w", encoding="utf-8") as f:
        json.dump(_daily_json(predict_2024_df, prediction_col="predicted_blacklist"), f, ensure_ascii=False, indent=2)

    with predict_label_json_path.open("w", encoding="utf-8") as f:
        json.dump(_label_json(predict_2024_df), f, ensure_ascii=False, indent=2)

    joblib.dump(
        {
            "value_model": value_model,
            "label_threshold": float(args.label_threshold),
            "features": features,
        },
        model_path,
    )

    metrics.update(
        {
            "target": "ST_history_label.csv label value (regression only)",
            "train_start_date": str(train_start_date.date()),
            "train_end_date": str(train_end_date.date()),
            "predict_start_date": str(predict_start_date.date()),
            "predict_end_date": str(predict_end_date.date()),
            "label_threshold": float(args.label_threshold),
            "train_label_rows_from_file": int(len(labels)),
            "prediction_only_rows": int(prediction_only_mask.sum()),
            "features": features,
            "history_daily_json": str(history_daily_path),
            "predict_2024_daily_json": str(predict_daily_path),
            "history_interval_json": str(history_interval_path),
            "predict_2024_interval_json": str(predict_interval_path),
            "blacklist_daily_json": str(blacklist_daily_path),
            "predict_2024_label_json": str(predict_label_json_path),
            "model": str(model_path),
        }
    )
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    df[
        [
            "stock_code",
            "date",
            "label_in_file",
            "y_label_empty",
            "y_label_value",
            "predicted_st",
            "predicted_blacklist",
            "predicted_label_empty",
            "predicted_label_value",
            "predicted_label",
            *RISK_COLUMNS,
        ]
    ].to_csv(prediction_csv_path, index=False)

    print("完成")
    print(f"history_daily_json={history_daily_path}")
    print(f"predict_2024_daily_json={predict_daily_path}")
    print(f"history_interval_json={history_interval_path}")
    print(f"predict_2024_interval_json={predict_interval_path}")
    print(f"blacklist_daily_json={blacklist_daily_path}")
    print(f"predict_2024_label_json={predict_label_json_path}")
    print(f"metrics={metrics_path}")
    print(f"model={model_path}")
    print(f"prediction_csv={prediction_csv_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a daily ST label regression model from LLM risk scores.")
    parser.add_argument("--risk-score-csv", default=str(RISK_SCORE_CSV))
    parser.add_argument("--st-label-csv", default=str(ST_URGENCY_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--train-start-date", default="2020-01-01")
    parser.add_argument("--train-end-date", default="2023-12-31")
    parser.add_argument("--predict-start-date", default="2024-01-01")
    parser.add_argument("--predict-end-date", default="2024-12-31")
    parser.add_argument("--validation-start", default="2023-01-01")
    parser.add_argument("--zero-ratio", type=float, default=5.537670)
    parser.add_argument("--label-threshold", type=float, default=0.017067)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=96)
    parser.add_argument("--history-daily-json-name", default="daily_st_status_2020_2023.json")
    parser.add_argument("--predict-2024-daily-json-name", default="daily_st_status_2024_predicted.json")
    parser.add_argument("--history-interval-json-name", default="st_status_intervals_2020_2023.json")
    parser.add_argument("--predict-2024-interval-json-name", default="st_status_intervals_2024_predicted.json")
    parser.add_argument("--blacklist-daily-json-name", default="daily_blacklist_predicted.json")
    parser.add_argument("--predict-2024-label-json-name", default="st_label_2024_predicted.json")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
