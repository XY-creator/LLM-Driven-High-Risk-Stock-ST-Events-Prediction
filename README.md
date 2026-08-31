# 茂源量化 LLM 预测 ST
## PlanA

1. `main.py`

   完整 LLM pipeline。会从公告文本开始做推理、生成风险分，再输出 blacklist。

2. `run_planA_with_4d_score.py`

   保留四维风险评估分数后直接读取，基于四维分的 max aggregation 生成 blacklist 和评估结果。

## PlanB

1. `auto_tune.py`

   参数调优模块。

2. 模型训练模块启动指令

   下面命令使用当前项目内的相对路径。默认输入文件放在 `PlanB/` 目录下，输出写入 `PlanB/output/st_status_model/`。

```bash
cd PlanB

python3 st_status_model.py \
  --risk-score-csv four_dimension_risk_score_new.csv \
  --st-label-csv ST_history_label.csv \
  --output-dir output/st_status_model \
  --train-start-date 2020-01-01 \
  --train-end-date 2022-12-31 \
  --predict-start-date 2023-01-01 \
  --predict-end-date 2023-12-31 \
  --validation-start 2022-01-01 \
  --negative-ratio 4.0 \
  --zero-ratio 4.0 \
  --label-threshold 0.002739726 \
  --random-state 42 \
  --max-iter 180 \
  --history-daily-json-name daily_st_status_2020_2022.json \
  --predict-2024-daily-json-name daily_st_status_2023_predicted.json \
  --history-interval-json-name st_status_intervals_2020_2022.json \
  --predict-2024-interval-json-name st_status_intervals_2023_predicted.json \
  --blacklist-daily-json-name daily_blacklist_predicted.json \
  --predict-2024-label-json-name st_label_2023_predicted.json
```
