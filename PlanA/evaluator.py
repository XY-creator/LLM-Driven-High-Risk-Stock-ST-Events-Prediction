# evaluator

import pandas as pd
import numpy as np

import config
from utils import _reshape_panel_for_daily
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(it, **kwargs):
        return it

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