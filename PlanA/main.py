# main

import json
from pathlib import Path
import multiprocessing as mp
import pandas as pd

import config
import dataset_builder
import inference
import evaluator
import utils

def main():
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    print("===== 构建数据（仅 output+lookback 范围） =====")

    start_date = (
        pd.Timestamp(config.OUTPUT_START_DATE)
        - pd.Timedelta(days=config.LOOKBACK_DAYS)
    ).strftime("%Y-%m-%d")
    end_date = config.OUTPUT_END_DATE

    data, calendar, universe, ann_daily = dataset_builder.build_daily_xy(
        start_date=start_date,
        end_date=end_date,
    )

    print("===== LLM 多GPU推理 + 生成两份输出 JSON =====")
    score_all, meta, out_blacklist, out_with_reason = inference.llm_infer_score_and_build_outputs(
        data=data,
        calendar=calendar,
        universe=universe,
        ann_daily=ann_daily,
    )

    print("===== 在区间评估指标 =====")
    metrics, _daily_df = evaluator.evaluate_screenshot_metrics_threshold(
        data=data,
        score=score_all,
        threshold=config.BLACKLIST_RETAIN_THRESHOLD,
        topk_cap=(config.BLACKLIST_TOPK if config.BLACKLIST_USE_TOPK_CAP else None),
    )
    meta["output"]["metrics"] = metrics

    today = pd.Timestamp.today().strftime("%Y%m%d")
    out_dir = Path(config.PREDICT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    submit_path = out_dir / f"{config.SUBMIT_PREFIX}_{today}.json"
    reason_path = out_dir / f"{config.SUBMIT_PREFIX}_{today}_reason.json"

    with open(submit_path, "w", encoding="utf-8") as f:
        json.dump(out_blacklist, f, ensure_ascii=False, indent=2)

    with open(reason_path, "w", encoding="utf-8") as f:
        json.dump(out_with_reason, f, ensure_ascii=False, indent=2)

    meta_path = utils.save_llm_meta(meta)

    print("===== 完成 =====")
    print("submit:", submit_path)
    print("reason:", reason_path)
    print("meta:", meta_path)
    print("metrics:", metrics)

if __name__ == "__main__":
    main()