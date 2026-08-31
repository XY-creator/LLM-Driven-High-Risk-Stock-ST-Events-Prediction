# utils

import json
import pandas as pd
import numpy as np
from pathlib import Path

import config

def _reshape_panel_for_daily(data, score, extra_series_dict=None):
    if "y_status" not in data.columns:
        raise ValueError("data 必须包含 y_status 列。")

    if not score.index.equals(data.index):
        score = score.reindex(data.index)

    df = pd.DataFrame(
        {
            "y_status": data["y_status"].astype(np.int8),
            "score": score.astype(np.float32),
        }
    )

    extra_series_dict = extra_series_dict or {}
    for k, v in extra_series_dict.items():
        if isinstance(v, pd.DataFrame):
            if k in v.columns:
                v = v[k]
            elif v.shape[1] == 1:
                v = v.iloc[:, 0]
            else:
                raise ValueError(f"extra df {k} 必须只有一列或包含同名列。")

        if not v.index.equals(data.index):
            v = v.reindex(data.index)

        df[k] = v.astype(object)

    df = df.sort_index(level=["stock_code", "date"])

    codes = df.index.get_level_values("stock_code").to_numpy()
    if len(codes) == 0:
        raise ValueError("data 为空，无法 reshape。")

    dates = pd.to_datetime(df.index.get_level_values("date").to_numpy()).values.astype("datetime64[ns]")

    first = codes[0]
    diff_pos = np.flatnonzero(codes != first)
    D = int(diff_pos[0]) if len(diff_pos) else int(len(codes))
    if D <= 0:
        raise ValueError("推断得到的 D<=0，index 结构异常。")

    U = int(len(codes) // D)
    if U * D != len(codes):
        raise ValueError(f"index 不是完整面板：len={len(codes)} 不能整除 D={D}。")

    codes_mat = codes.reshape(U, D)
    dates_mat = dates.reshape(U, D)

    code_vec = codes_mat[:, 0]
    date_vec = pd.DatetimeIndex(dates_mat[0]).normalize()

    y_status_m = df["y_status"].to_numpy(np.int8).reshape(U, D)
    score_m = df["score"].to_numpy(np.float32).reshape(U, D)
    score_m = np.where(np.isnan(score_m), -np.inf, score_m)

    extra_m = {}
    for k in extra_series_dict.keys():
        extra_m[k] = df[k].to_numpy(dtype=object).reshape(U, D)

    return code_vec, date_vec, y_status_m, score_m, extra_m

def save_llm_meta(meta):
    out_dir = Path(config.ARTIFACT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dir = meta.get("run_dir", None)
    if run_dir:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        meta_path_run = run_dir / "meta.json"
        with open(meta_path_run, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    meta_path = out_dir / "llm_risk_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return str(meta_path)