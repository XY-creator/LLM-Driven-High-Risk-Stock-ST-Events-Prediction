import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np

import config
from utils import _reshape_panel_for_daily
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(it, **kwargs):
        return it


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RISK_SCORE_CSV = BASE_DIR / "four_dimension_risk_score.csv"
DEFAULT_ST_LABEL_CSV = BASE_DIR / "ST_history_label.csv"
DEFAULT_BLACKLIST_JSON = BASE_DIR / "st_status_model" / "daily_blacklist_predicted.json"


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


def _read_st_label_truth(path, start_date, end_date):
    wide = pd.read_csv(path, low_memory=False)
    date_col = wide.columns[0]
    wide[date_col] = pd.to_datetime(wide[date_col], errors="coerce").dt.normalize()
    wide = wide[(wide[date_col] >= start_date) & (wide[date_col] <= end_date)].copy()

    truth = wide.melt(id_vars=date_col, var_name="stock_code", value_name="st_label")
    truth = truth.rename(columns={date_col: "date"})
    truth["stock_code"] = truth["stock_code"].map(_normalize_code)
    truth["label_in_file"] = np.int8(1)
    truth["y_status"] = truth["st_label"].isna().astype(np.int8)
    truth["st_label_value"] = pd.to_numeric(truth["st_label"], errors="coerce").clip(0, 1)
    return truth[["date", "stock_code", "label_in_file", "y_status", "st_label_value"]]


def _add_st_label_truth(df, truth):
    df = df.merge(truth, on=["date", "stock_code"], how="left")
    df["label_in_file"] = df["label_in_file"].fillna(0).astype(np.int8)
    df["y_status"] = df["y_status"].fillna(0).astype(np.int8)
    df.loc[df["label_in_file"] == 0, "st_label_value"] = 0.0
    return df


def _daily_json_to_frame(path, start_date, end_date):
    with open(path, "r", encoding="utf-8") as f:
        daily = json.load(f)

    rows = []
    for date_text, codes in daily.items():
        date = pd.to_datetime(date_text, errors="coerce")
        if pd.isna(date):
            continue
        date = date.normalize()
        if date < start_date or date > end_date:
            continue
        for code in codes:
            rows.append((date, _normalize_code(code), 1))

    if not rows:
        return pd.DataFrame(columns=["date", "stock_code", "predicted_blacklist"])
    pred = pd.DataFrame(rows, columns=["date", "stock_code", "predicted_blacklist"])
    pred = pred.drop_duplicates(["date", "stock_code"])
    return pred


def evaluate_blacklist_json(
    blacklist_json_path=DEFAULT_BLACKLIST_JSON,
    risk_score_csv=DEFAULT_RISK_SCORE_CSV,
    st_label_csv=DEFAULT_ST_LABEL_CSV,
    start_date="2020-01-01",
    end_date="2023-12-31",
    output_dir=None,
):
    start_date = pd.Timestamp(start_date).normalize()
    end_date = pd.Timestamp(end_date).normalize()

    universe = pd.read_csv(
        risk_score_csv,
        usecols=["stock_code", "date"],
        dtype={"stock_code": "string"},
        parse_dates=["date"],
        low_memory=False,
    )
    universe["date"] = pd.to_datetime(universe["date"], errors="coerce").dt.normalize()
    universe = universe.dropna(subset=["stock_code", "date"])
    universe = universe[(universe["date"] >= start_date) & (universe["date"] <= end_date)].copy()
    universe["stock_code"] = universe["stock_code"].map(_normalize_code)
    universe = universe.drop_duplicates(["date", "stock_code"]).sort_values(["stock_code", "date"])

    truth = _read_st_label_truth(st_label_csv, start_date, end_date)
    universe = _add_st_label_truth(universe, truth)

    pred = _daily_json_to_frame(blacklist_json_path, start_date, end_date)
    eval_df = universe.merge(pred, on=["date", "stock_code"], how="left")
    eval_df["predicted_blacklist"] = eval_df["predicted_blacklist"].fillna(0).astype(np.int8)
    eval_df = eval_df.sort_values(["stock_code", "date"])

    codes = eval_df["stock_code"].to_numpy()
    dates = eval_df["date"].to_numpy("datetime64[ns]")
    if len(codes) == 0:
        raise ValueError("评估面板为空。")
    first = codes[0]
    diff_pos = np.flatnonzero(codes != first)
    d_count = int(diff_pos[0]) if len(diff_pos) else int(len(codes))
    u_count = int(len(codes) // d_count)
    if u_count * d_count != len(codes):
        raise ValueError("评估面板不是完整 stock_code x date 面板，无法按原评估逻辑 reshape。")

    code_vec = codes.reshape(u_count, d_count)[:, 0]
    date_vec = pd.DatetimeIndex(dates.reshape(u_count, d_count)[0]).normalize()
    y_status_m = eval_df["y_status"].to_numpy(np.int8).reshape(u_count, d_count)
    label_m = eval_df["st_label_value"].to_numpy(np.float32).reshape(u_count, d_count)
    in_bl = eval_df["predicted_blacklist"].to_numpy(bool).reshape(u_count, d_count)

    in_bl_eff = in_bl & (y_status_m == 0)
    prev = np.concatenate([y_status_m[:, [0]], y_status_m[:, :-1]], axis=1)
    entry = (y_status_m == 1) & (prev == 0)

    horizon = int(config.HORIZON_DAYS)
    min_lead = int(config.MIN_LEAD_DAYS)
    min_label = 1.0 / horizon if horizon > 0 else np.inf
    max_label = 1.0 / min_lead if min_lead > 0 else np.inf
    correct_day = in_bl_eff & (label_m >= min_label) & (label_m <= max_label)
    predicted_event = np.zeros((u_count, d_count), dtype=bool)

    for i in tqdm(range(u_count), desc="评估: 逐股匹配事件", unit="stock"):
        for event_pos in np.flatnonzero(entry[i]):
            left = max(0, event_pos - horizon)
            right = max(0, event_pos - min_lead + 1)
            if right <= left:
                continue
            if in_bl_eff[i, left:right].any():
                predicted_event[i, event_pos] = True

    bl_cnt = in_bl_eff.sum(axis=0).astype(np.int32)
    correct_cnt = correct_day.sum(axis=0).astype(np.int32)
    st_cnt = (y_status_m == 1).sum(axis=0).astype(np.int32)

    daily_acc = np.where(bl_cnt > 0, correct_cnt / bl_cnt, np.nan).astype(np.float32)
    fp_cnt = (bl_cnt - correct_cnt).astype(np.int32)
    denom = (int(config.MARKET_N) - st_cnt - correct_cnt).astype(np.int64)
    daily_fpr = np.where(denom > 0, fp_cnt / denom, np.nan).astype(np.float32)

    acc = float(np.nanmean(daily_acc)) if np.isfinite(daily_acc).any() else None
    fpr = float(np.nanmean(daily_fpr)) if np.isfinite(daily_fpr).any() else None
    total_events = int(entry.sum())
    hit_events = int((predicted_event & entry).sum())
    recall = (hit_events / total_events) if total_events > 0 else None

    daily_df = pd.DataFrame(
        {
            "date": date_vec,
            "blacklist_cnt": bl_cnt,
            "correct_cnt": correct_cnt,
            "daily_acc": daily_acc,
            "daily_fpr": daily_fpr,
            "st_cnt": st_cnt,
        }
    )

    metrics = {
        "task": "blacklist_early_warning",
        "Acc": acc,
        "FPR": fpr,
        "Recall": recall,
        "total_events": total_events,
        "hit_events": hit_events,
        "avg_blacklist_size": float(np.nanmean(bl_cnt)),
        "market_n": int(config.MARKET_N),
        "horizon_days": horizon,
        "min_lead_days": min_lead,
        "start_date": str(start_date.date()),
        "end_date": str(end_date.date()),
        "blacklist_json_path": str(blacklist_json_path),
        "risk_score_csv": str(risk_score_csv),
        "st_label_csv": str(st_label_csv),
        "universe_size": int(len(code_vec)),
        "total_days": int(len(date_vec)),
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "blacklist_json_eval_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        daily_df.to_csv(output_dir / "blacklist_json_eval_by_day.csv", index=False)

    return metrics, daily_df

def evaluate_screenshot_metrics_threshold(data, score, threshold, topk_cap=None):
    if "y_status" not in data.columns:
        raise ValueError("data 必须包含 y_status 列。")

    code_vec, date_vec, y_status_m, score_m, _ = _reshape_panel_for_daily(data, score)

    U = int(len(code_vec))
    D = int(len(date_vec))

    out_start = pd.Timestamp(config.OUTPUT_START_DATE).normalize()
    out_end = pd.Timestamp(config.OUTPUT_END_DATE).normalize()
    col_out = (date_vec >= out_start) & (date_vec <= out_end)

    thr = float(threshold)

    cap = None
    if topk_cap is not None:
        cap = max(0, min(int(topk_cap), U))

    in_bl = np.zeros((U, D), dtype=bool)
    for j in tqdm(range(D), desc="评估: 逐日构建黑名单", unit="day"):
        elig = np.flatnonzero(score_m[:, j] >= thr)
        if len(elig) == 0:
            continue

        if cap is not None and len(elig) > cap:
            sc = score_m[elig, j]
            picks = elig[np.argpartition(-sc, cap - 1)[:cap]]
            in_bl[picks, j] = True
        else:
            in_bl[elig, j] = True

    in_bl_eff = in_bl & (y_status_m == 0)

    prev = np.concatenate([y_status_m[:, [0]], y_status_m[:, :-1]], axis=1)
    entry = (y_status_m == 1) & (prev == 0)

    correct_day = np.zeros((U, D), dtype=bool)
    predicted_event = np.zeros((U, D), dtype=bool)

    horizon = int(config.HORIZON_DAYS)
    min_lead = int(config.MIN_LEAD_DAYS)

    for i in tqdm(range(U), desc="评估: 逐股匹配事件", unit="stock"):
        bl = in_bl_eff[i]
        if not bl.any():
            continue

        prev_bl = np.r_[False, bl[:-1]]
        next_bl = np.r_[bl[1:], False]
        starts = np.flatnonzero(bl & ~prev_bl)
        ends = np.flatnonzero(bl & ~next_bl)

        ent_i = entry[i]

        for s, e in zip(starts, ends):
            if e >= D - 1:
                continue

            event_pos = e + 1
            if not ent_i[event_pos]:
                continue

            lead = event_pos - s
            if lead < min_lead or lead > horizon:
                continue

            if bool(config.ONLY_USE_EVENTS_WITHIN_OUTPUT):
                d_event = date_vec[event_pos]
                if not (out_start <= d_event <= out_end):
                    continue

            correct_day[i, s : e + 1] = True
            predicted_event[i, event_pos] = True

    bl_cnt = in_bl_eff[:, col_out].sum(axis=0).astype(np.int32)
    correct_cnt = correct_day[:, col_out].sum(axis=0).astype(np.int32)
    st_cnt = (y_status_m[:, col_out] == 1).sum(axis=0).astype(np.int32)

    daily_acc = np.where(bl_cnt > 0, correct_cnt / bl_cnt, np.nan).astype(np.float32)

    fp_cnt = (bl_cnt - correct_cnt).astype(np.int32)
    denom = (int(config.MARKET_N) - st_cnt - correct_cnt).astype(np.int64)
    daily_fpr = np.where(denom > 0, fp_cnt / denom, np.nan).astype(np.float32)

    acc = float(np.nanmean(daily_acc))
    fpr = float(np.nanmean(daily_fpr))

    total_events = int(entry[:, col_out].sum())
    hit_events = int((predicted_event & entry)[:, col_out].sum())
    recall = (hit_events / total_events) if total_events > 0 else np.nan

    daily_df = pd.DataFrame(
        {
            "date": date_vec[col_out],
            "blacklist_cnt": bl_cnt,
            "correct_cnt": correct_cnt,
            "daily_acc": daily_acc,
            "daily_fpr": daily_fpr,
            "st_cnt": st_cnt,
        }
    ).set_index("date")

    metrics = {
        "Acc": acc,
        "FPR": fpr,
        "Recall": (float(recall) if recall == recall else None),
        "total_events": total_events,
        "hit_events": hit_events,
        "avg_blacklist_size": float(np.nanmean(bl_cnt)),
        "threshold": float(thr),
        "topk_cap": (int(cap) if cap is not None else None),
        "market_n": int(config.MARKET_N),
        "horizon_days": horizon,
        "min_lead_days": min_lead,
        "only_use_events_within_output": bool(config.ONLY_USE_EVENTS_WITHIN_OUTPUT),
        "output_start": config.OUTPUT_START_DATE,
        "output_end": config.OUTPUT_END_DATE,
    }
    return metrics, daily_df


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate blacklist JSON by the task definition.")
    parser.add_argument("--blacklist-json", default=str(DEFAULT_BLACKLIST_JSON))
    parser.add_argument("--risk-score-csv", default=str(DEFAULT_RISK_SCORE_CSV))
    parser.add_argument("--st-label-csv", default=str(DEFAULT_ST_LABEL_CSV))
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--end-date", default="2023-12-31")
    parser.add_argument("--output-dir", default=str(BASE_DIR / "output" / "st_status_model"))
    return parser.parse_args()


def main():
    args = parse_args()
    metrics, _ = evaluate_blacklist_json(
        blacklist_json_path=args.blacklist_json,
        risk_score_csv=args.risk_score_csv,
        st_label_csv=args.st_label_csv,
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output_dir,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()