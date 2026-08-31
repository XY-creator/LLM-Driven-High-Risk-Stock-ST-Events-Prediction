# main

from pathlib import Path
import multiprocessing as mp
import pandas as pd

import config
import dataset_builder
import inference
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

    print("===== LLM 多GPU推理 & 四维度打分 =====")
    score_df, meta = inference.llm_infer_score_and_build_outputs(
        data=data,
        calendar=calendar,
        universe=universe,
        ann_daily=ann_daily,
    )

    # 截取最终需要输出结果的时间段
    out_start = pd.Timestamp(config.OUTPUT_START_DATE).normalize()
    out_end = pd.Timestamp(config.OUTPUT_END_DATE).normalize()
    dates = pd.DatetimeIndex(score_df.index.get_level_values("date"))
    final_score_df = score_df[(dates >= out_start) & (dates <= out_end)]

    # 结果输出到项目根目录下的 CSV 文件中
    out_dir = Path(config.PREDICT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 按照需求保存为双 index 的 csv
    csv_path = out_dir / "four_dimension_risk_score.csv"
    final_score_df.to_csv(csv_path)

    # 保存元数据以便以后溯源
    meta_path = utils.save_llm_meta(meta)

    print("===== 完成 =====")
    print("CSV 已保存至:", csv_path)
    print("Meta 信息保存至:", meta_path)

if __name__ == "__main__":
    main()