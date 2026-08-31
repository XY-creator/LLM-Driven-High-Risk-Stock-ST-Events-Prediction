# inference

import pandas as pd
import numpy as np
from pathlib import Path

import config
import llm_engine
import fusion_strategy
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(it, **kwargs):
        return it

# 四个维度的列名定义
SCORE_COLS = ["financial_risk", "normative_risk", "illegal_risk", "other_risk"]

def _normalize_ann_series(ann_daily):
    if isinstance(ann_daily, pd.DataFrame):
        if "ann_daily" in ann_daily.columns:
            return ann_daily["ann_daily"]
        if ann_daily.shape[1] == 1:
            return ann_daily.iloc[:, 0]
        raise ValueError("ann_daily DataFrame 必须包含 'ann_daily' 列或仅一列。")
    return ann_daily


def _run_llm_for_mask(data, infer_mask, text_series=None):
    if text_series is None:
        text_series = data["ann_text"]
    else:
        if not text_series.index.equals(data.index):
            text_series = text_series.reindex(data.index)

    infer_df = pd.DataFrame({"ann_text": text_series.loc[infer_mask].astype(str)})
    infer_df = infer_df.sort_index(level=["stock_code", "date"])

    idx = infer_df.index
    stock_codes = idx.get_level_values("stock_code").astype(str).to_list()
    date_strs = pd.DatetimeIndex(idx.get_level_values("date")).normalize().strftime("%Y-%m-%d").to_list()
    ann_texts = infer_df["ann_text"].astype(str).to_list()

    task_records = []
    for sc, ds, at in tqdm(
        zip(stock_codes, date_strs, ann_texts),
        total=len(stock_codes),
        desc="组装LLM任务",
        unit="sample",
    ):
        task_records.append({"stock_code": sc, "date": ds, "ann_text": at})

    run_id = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.LLM_RUNS_DIR) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    results, infer_meta = llm_engine.parallel_infer_risk_scores(
        task_records=task_records,
        run_dir=run_dir,
        gpu_ids=None,
        batch_size=int(getattr(config, "LLM_BATCH_SIZE", 4)),
    )

    return results, infer_meta, run_id, run_dir, idx, task_records


def _build_raw_score_from_results(data, results):
    """
    将 LLM 输出的 dict 转换为 4维 DataFrame
    """
    score_df = pd.DataFrame(np.nan, index=data.index, columns=SCORE_COLS, dtype=np.float32)

    if results:
        res_codes = [r["stock_code"] for r in results]
        res_dates = [pd.Timestamp(r["date"]).normalize() for r in results]
        res_index = pd.MultiIndex.from_arrays([res_codes, res_dates], names=["stock_code", "date"])
        
        # 构建一个临时的 4维 DF
        res_data = {c: [float(r.get(c, 0)) for r in results] for c in SCORE_COLS}
        temp_df = pd.DataFrame(res_data, index=res_index, dtype=np.float32)
        
        # 确保去重并放入对应的 index 中
        temp_df = temp_df[~temp_df.index.duplicated(keep='last')]
        temp_df = temp_df.reindex(data.index)
        
        # 仅更新非 NaN 的部分
        score_df.update(temp_df)

    return score_df


def _build_common_meta(data, output_mask, run_id, run_dir, infer_meta, task_records):
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "llm_config": {
            "model_path": str(config.LLM_MODEL_PATH),
            "load_in_4bit": bool(getattr(config, "LLM_LOAD_IN_4BIT", False)),
            "max_input_tokens": int(getattr(config, "LLM_MAX_INPUT_TOKENS", 1024)),
            "max_new_tokens": int(getattr(config, "LLM_MAX_NEW_TOKENS", 200)),
            "batch_size": int(getattr(config, "LLM_BATCH_SIZE", 4)),
            "infer_only_on_ann_days": bool(getattr(config, "LLM_INFER_ONLY_ON_ANN_DAYS", True)),
            "bootstrap_on_output_start": bool(getattr(config, "LLM_BOOTSTRAP_ON_OUTPUT_START", True)),
            "skip_empty_ann_text": bool(getattr(config, "LLM_SKIP_EMPTY_ANN_TEXT", True)),
            "lookback_days": int(getattr(config, "LOOKBACK_DAYS", 60)),
            "max_ann_per_day": int(getattr(config, "MAX_ANN_PER_DAY", 3)),
            "use_fulltext_key_sentences": bool(getattr(config, "USE_FULLTEXT_KEY_SENTENCES", True)),
        },
        "infer_period": {
            "output_start": config.OUTPUT_START_DATE,
            "output_end": config.OUTPUT_END_DATE,
        },
        "infer_stats": {
            "n_infer_points": int(len(task_records)),
            "n_output_points": int(output_mask.sum()),
            "n_universe": int(len(set(data.index.get_level_values("stock_code")))),
        },
        "parallel_infer": infer_meta,
        "output": {},
    }

def llm_infer_and_score_scheme_a(data, ann_daily):
    ann_s = _normalize_ann_series(ann_daily)

    if not ann_s.index.equals(data.index):
        ann_s = ann_s.reindex(data.index)

    dates = pd.DatetimeIndex(data.index.get_level_values("date")).normalize()
    out_start = pd.Timestamp(config.OUTPUT_START_DATE).normalize()
    out_end = pd.Timestamp(config.OUTPUT_END_DATE).normalize()
    output_mask = (dates >= out_start) & (dates <= out_end)

    is_ann_day = ann_s.astype(str) != "无相关公告"
    infer_mask = output_mask.copy()

    if bool(getattr(config, "LLM_INFER_ONLY_ON_ANN_DAYS", True)):
        infer_mask &= is_ann_day

    if bool(getattr(config, "LLM_BOOTSTRAP_ON_OUTPUT_START", True)):
        infer_mask |= (dates == out_start)

    if bool(getattr(config, "LLM_SKIP_EMPTY_ANN_TEXT", True)):
        infer_mask &= (data["ann_text"].astype(str) != "无相关公告")

    results, infer_meta, run_id, run_dir, idx, task_records = _run_llm_for_mask(data, infer_mask)
    score_df = _build_raw_score_from_results(data, results)
    
    # 填充非输出区间为0
    score_df.loc[~output_mask, SCORE_COLS] = 0.0

    post = {
        "score_ffill": bool(getattr(config, "LLM_SCORE_FFILL", True)),
        "ffill_max_days": getattr(config, "LLM_FFILL_MAX_DAYS", 30),
    }

    if bool(getattr(config, "LLM_SCORE_FFILL", True)):
        out_idx = score_df.index[output_mask]
        out_df = score_df.loc[out_idx].copy()

        out_df = out_df.sort_index(level=["stock_code", "date"])
        out_df = out_df.groupby(level="stock_code").ffill()

        score_df.loc[out_idx] = out_df
        score_df.loc[out_idx] = score_df.loc[out_idx].fillna(0.0)

        max_days = getattr(config, "LLM_FFILL_MAX_DAYS", 30)
        if max_days is not None:
            max_days = int(max_days)

            infer_pos = pd.Index(idx).intersection(out_idx)
            out_dates = pd.DatetimeIndex(out_idx.get_level_values("date")).normalize()
            out_dates_s = pd.Series(out_dates.to_numpy(), index=out_idx)

            infer_date_s = pd.Series(pd.NaT, index=out_idx, dtype="datetime64[ns]")
            infer_date_s.loc[infer_pos] = out_dates_s.loc[infer_pos]

            last_infer = infer_date_s.sort_index(level=["stock_code", "date"]).groupby(level="stock_code").ffill()
            days_since = (out_dates_s - last_infer).dt.days

            stale = days_since > max_days
            if stale.any():
                score_df.loc[stale[stale].index, SCORE_COLS] = 0.0

    meta = _build_common_meta(data, output_mask, run_id, run_dir, infer_meta, task_records)
    meta["infer_scheme"] = "scheme_a"
    meta["post_process"] = post

    return score_df, meta


def llm_infer_and_score_fusion_decay(data, ann_daily):
    ann_s = _normalize_ann_series(ann_daily)

    if not ann_s.index.equals(data.index):
        ann_s = ann_s.reindex(data.index)

    dates = pd.DatetimeIndex(data.index.get_level_values("date")).normalize()
    out_start = pd.Timestamp(config.OUTPUT_START_DATE).normalize()
    out_end = pd.Timestamp(config.OUTPUT_END_DATE).normalize()
    output_mask = (dates >= out_start) & (dates <= out_end)
    memory_mask = dates <= out_end

    is_ann_day = ann_s.astype(str) != "无相关公告"
    infer_mask = memory_mask.copy()

    if bool(getattr(config, "LLM_INFER_ONLY_ON_ANN_DAYS", True)):
        infer_mask &= is_ann_day

    if bool(getattr(config, "LLM_SKIP_EMPTY_ANN_TEXT", True)):
        infer_mask &= (data["ann_text"].astype(str) != "无相关公告")

    prefilter_meta = {
        "enabled": bool(getattr(config, "FUSION_PREFILTER_ENABLE", True)),
    }

    # 预筛选逻辑基于公告内容粗筛，不涉及四维度定义，保留原样
    if bool(getattr(config, "FUSION_PREFILTER_ENABLE", True)):
        prefilter = fusion_strategy.AdvancedFastFilter(
            keywords=list(getattr(config, "RISK_KEYWORDS", [])),
            high_th=int(getattr(config, "FUSION_FILTER_HIGH_TH", 4)),
            medium_th=int(getattr(config, "FUSION_FILTER_MEDIUM_TH", 2)),
        )
        flt_df = prefilter.classify_series(ann_s)
        allowed_labels = tuple(getattr(config, "FUSION_ALLOWED_LABELS", ("HIGH", "MEDIUM")))
        allow = flt_df["filter_label"].isin(allowed_labels)
        infer_mask &= allow

        prefilter_meta.update(
            {
                "high_th": int(getattr(config, "FUSION_FILTER_HIGH_TH", 4)),
                "medium_th": int(getattr(config, "FUSION_FILTER_MEDIUM_TH", 2)),
                "allowed_labels": list(allowed_labels),
                "n_high": int((flt_df["filter_label"] == "HIGH").sum()),
                "n_medium": int((flt_df["filter_label"] == "MEDIUM").sum()),
                "n_low": int((flt_df["filter_label"] == "LOW").sum()),
                "n_allowed_points": int(allow.sum()),
            }
        )

    if bool(getattr(config, "LLM_BOOTSTRAP_ON_OUTPUT_START", True)):
        infer_mask |= (dates == out_start)

    use_daily_text = bool(getattr(config, "FUSION_USE_DAILY_TEXT", True))
    infer_text = ann_s if use_daily_text else data["ann_text"]
    
    results, infer_meta, run_id, run_dir, _, task_records = _run_llm_for_mask(
        data,
        infer_mask,
        text_series=infer_text,
    )
    
    # 获得 4维 DataFrame
    raw_score_df = _build_raw_score_from_results(data, results)

    decay = float(getattr(config, "FUSION_DECAY_PER_DAY", 0.5))
    # 可以给四维设置同一个阈值进入记忆（如果没有指定，沿用旧的阈值配置）
    threshold = float(getattr(config, "RISK_SCORE_THRESHOLD", 0.0)) # 设为0表示四维度均允许衰减，你可以在config中重新定义

    # 对四个维度分别应用记忆衰减机制
    score_df = fusion_strategy.apply_decay_memory_scores(
        raw_score_df=raw_score_df,
        output_mask=output_mask,
        decay_per_day=decay,
        threshold=threshold,
        memory_mask=memory_mask,
    )

    meta = _build_common_meta(data, output_mask, run_id, run_dir, infer_meta, task_records)
    meta["infer_scheme"] = "fusion_decay"
    meta["fusion"] = {
        "decay_per_day": decay,
        "memory_threshold": threshold,
        "use_daily_text": use_daily_text,
        "memory_warmup_start": str(dates[memory_mask].min().date()) if memory_mask.any() else None,
        "prefilter": prefilter_meta,
    }
    meta["post_process"] = {
        "mode": "baseline_style_decay_memory",
        "score_ffill": False,
        "ffill_max_days": None,
    }

    return score_df, meta

def llm_infer_score_and_build_outputs(data, calendar, universe, ann_daily):
    scheme = str(getattr(config, "LLM_INFER_SCHEME", "scheme_a")).strip().lower()
    if scheme == "fusion_decay":
        score_df, meta = llm_infer_and_score_fusion_decay(data, ann_daily=ann_daily)
    else:
        score_df, meta = llm_infer_and_score_scheme_a(data, ann_daily=ann_daily)

    # 移除黑名单逻辑，直接返回多维度分数
    return score_df, meta
