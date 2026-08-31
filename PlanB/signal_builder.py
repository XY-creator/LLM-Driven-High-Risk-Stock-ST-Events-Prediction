# signal_builder

import pandas as pd
import numpy as np
import config
from utils import _reshape_panel_for_daily
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(it, **kwargs):
        return it

def build_blacklist_json_threshold(data, score, threshold, topk_cap=None):
    code_vec, date_vec, y_status_m, score_m, _ = _reshape_panel_for_daily(data, score)

    out_start = pd.Timestamp(config.OUTPUT_START_DATE).normalize()
    out_end = pd.Timestamp(config.OUTPUT_END_DATE).normalize()
    col_out = (date_vec >= out_start) & (date_vec <= out_end)

    thr = float(threshold)
    U = int(len(code_vec))

    cap = None
    if topk_cap is not None:
        cap = max(0, min(int(topk_cap), U))

    out = {}
    out_cols = np.where(col_out)[0]
    for j in tqdm(out_cols, desc="生成黑名单JSON", unit="day"):
        elig = np.flatnonzero(score_m[:, j] >= thr)

        if cap is not None:
            if cap == 0 or len(elig) == 0:
                picks = np.array([], dtype=int)
            elif len(elig) > cap:
                sc = score_m[elig, j]
                picks = elig[np.argpartition(-sc, cap - 1)[:cap]]
            else:
                picks = elig
        else:
            picks = elig

        if len(picks) > 0:
            picks = picks[y_status_m[picks, j] == 0]

        if len(picks) > 1:
            codes_p = code_vec[picks].astype(str)
            scores_p = score_m[picks, j]
            order = np.lexsort((codes_p, -scores_p))
            picks = picks[order]

        out[str(date_vec[j].date())] = [str(x) for x in code_vec[picks]]

    return out

def build_with_reason_json_threshold(data, score, ann_daily, threshold, topk_cap=None):
    if isinstance(ann_daily, pd.DataFrame):
        if "ann_daily" in ann_daily.columns:
            ann_s = ann_daily["ann_daily"]
        elif ann_daily.shape[1] == 1:
            ann_s = ann_daily.iloc[:, 0]
        else:
            raise ValueError("ann_daily DataFrame 必须包含 'ann_daily' 列或仅一列。")
    else:
        ann_s = ann_daily

    code_vec, date_vec, y_status_m, score_m, extra_m = _reshape_panel_for_daily(
        data, score, extra_series_dict={"ann_daily": ann_s}
    )
    ann_m = extra_m["ann_daily"]

    out_start = pd.Timestamp(config.OUTPUT_START_DATE).normalize()
    out_end = pd.Timestamp(config.OUTPUT_END_DATE).normalize()
    col_out = (date_vec >= out_start) & (date_vec <= out_end)

    thr = float(threshold)
    U = int(len(code_vec))

    cap = None
    if topk_cap is not None:
        cap = max(0, min(int(topk_cap), U))

    out = {}
    out_cols = np.where(col_out)[0]
    for j in tqdm(out_cols, desc="生成带原因JSON", unit="day"):
        elig = np.flatnonzero(score_m[:, j] >= thr)

        if cap is not None:
            if cap == 0 or len(elig) == 0:
                picks = np.array([], dtype=int)
            elif len(elig) > cap:
                sc = score_m[elig, j]
                picks = elig[np.argpartition(-sc, cap - 1)[:cap]]
            else:
                picks = elig
        else:
            picks = elig

        if len(picks) > 0:
            picks = picks[y_status_m[picks, j] == 0]

        if len(picks) > 1:
            codes_p = code_vec[picks].astype(str)
            scores_p = score_m[picks, j]
            order = np.lexsort((codes_p, -scores_p))
            picks = picks[order]

        day_list = []
        for i in picks:
            rs = score_m[i, j]
            if not np.isfinite(rs):
                rs = 0

            txt = ann_m[i, j]
            if txt is None or (isinstance(txt, float) and np.isnan(txt)) or txt == "":
                txt = "无相关公告"

            day_list.append(
                {
                    "stock_code": str(code_vec[i]),
                    "risk_score": int(rs),
                    "is_st": int(y_status_m[i, j]),  
                    "ann_daily": str(txt),
                }
            )

        out[str(date_vec[j].date())] = day_list

    return out