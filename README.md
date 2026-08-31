# LLM-Based High-Risk Stock Blacklist Prediction System / 基于LLM的高风险股票黑名单预测系统

[![Language](https://img.shields.io/badge/Language-Python-blue.svg)](https://www.python.org/)
[![Model](https://img.shields.io/badge/Model-Qwen3--8B-orange.svg)]()
[![ML Framework](https://img.shields.io/badge/ML-Scikit--Learn-yellow.svg)](https://scikit-learn.org/)

*(English Version follows the Chinese Version)*

---

## 中文版 (Chinese Version)

### 项目背景与业务痛点
本项目旨在通过大语言模型 (LLM) 深度解析企业历史公告，提前预测A股市场的高风险股票，即可能被交易所实施“其他风险警示 (ST)”或“退市风险警示 (*ST)”的股票。

在量化交易和私募业务中，提前预测此类极端风险具有核心的商业价值。当股票发生ST事件时，传统量化模型往往面临以下毁灭性痛点：
1. **流动性枯竭与数据失真：** 极端风险导致盘面流动性丧失，量化交易完全失效，模型输出的信号失去执行意义。
2. **策略回撤：** 伴随退市等极端风险，股票净值暴跌可能造成巨大的本金损失。
3. **合规与系统硬拦截：** 触及高风险状态后，风控系统会进行硬拦截，导致量化交易指令根本无法下发。

同时，传统基于财务指标或浅层文本挖掘（非LLM方法）的预测模型，容易漏掉隐含的风险提示（如财务隐患、规范性风险、违法风险等），且难以捕捉需要复杂逻辑推理与全局上下文理解的风险信号。因此，本项目引入全模态大模型进行非结构化信息的深度特征提取，每日输出高风险股票“黑名单”，辅助风控系统在事件正式发生前进行主动调仓，避免非必要强制卖出与踏空。

### 核心方法与模型架构设计
本系统处理2019-2024年的A股公告数据，通过多阶段的自动化流程与双轨预警方案（Plan A 与 Plan B）实现高精度预警。

#### 1. LLM 结构化风险解析模块
系统采用全模态大模型 **Qwen3-8B**，对海量公告文本进行推理，输出四个维度的连续风险评分（0-100）：
* **财务风险 (Financial)：** 识别反映财务恶化与退市压力的信号，如净利润为负、营收低于3亿元、审计否定意见等。
* **规范风险 (Normative)：** 识别反映信息披露与治理缺陷的信号，如年报未披露、信息披露缺陷、资金占用等。
* **违法风险 (Illegal)：** 识别反映重大违法与监管处罚的信号，如财务造假、重大违法、欺诈发行等。
* **其他风险 (Other)：** 识别反映经营异常与流动性风险的信号，如银行账户冻结、违规担保、持续经营异常等。

#### 2. Plan A：规则驱动风险预警系统
Plan A 是一套基于专家规则与网格搜索的业务逻辑系统。
* **风险聚合设计：** 采用四维风险分中的**最大值 (Max)** 作为综合风险分数。此设计的动机在于，ST风险通常由单一极端高风险事件触发，因此极端风险维度的重要性远超平均风险水平。
* **时序衰减与双阈值机制：** ST风险具有持续性，公告影响不会在单日消失。系统设计了“风险记忆时序衰减机制”，结合进入阈值（Entry Threshold）和退出阈值（Retain Threshold）。当无新增高风险公告时，风险状态随时间衰减，直至低于退出阈值才移出黑名单。
* **参数优化：** 使用网格搜索 (Grid Search) 寻找不同成本预算下（如黑名单上限100/150）的最优参数，在召回率与名单规模之间达成平衡。

#### 3. Plan B：机器学习风险预警系统 (高级演进版)
Plan B 在 Plan A 的基础上，引入了监督学习框架进行二次建模。
* **连续型标签体系设计：** 传统做法采用 0/1 二分类（正常/ST），忽略了风险发酵的渐进性。本项目创新地将标签构建为距离下一次ST天数的倒数：`y_t = 1/(M-N)`（M为未来ST开始日，N为当前日期），使模型能够直接学习风险紧迫度的演变规律。
* **多维度特征工程：**
  * *时序衰减：* 四个维度的原始 LLM 评分独立进行每日衰减（如系数 0.5），新分数生成逻辑为 `max(旧值, 新值)`。
  * *横向聚合：* 提取每日风险最大值、非零维度数，反映风险的极端性与广泛性。
  * *纵向平滑：* 计算月度风险均值，以平滑单日噪音。
  * *周期性编码：* 使用 `sin(2πt/12)` 和 `cos(2πt/12)` 捕捉年报季等时间周期规律。
* **建模与调参：** 模型采用 `HistGradientBoostingRegressor` 回归器，原生支持缺失值与直方图加速。并使用 Optuna 贝叶斯优化算法，以 `Precision - FPR` 为目标函数进行自动超参数寻优。

### 评估体系与经济价值
ST预测任务存在极度不平衡的特性（全市场ST占比约1.2%）。在业务逻辑中，漏报的代价（平均跌幅超50%的本金损失）呈现极度不对称性，远高于误报的代价（年化5%的机会成本）。因此，系统的评估标准明确了**召回率的优先级远高于准确率**。

在2024年的测试集回测中：
* **性能表现：** Plan B 仅需要平均每日输出 9.3 只股票的黑名单（仅为 Plan A 规模的 1/12），便实现了高达 65.3% 的召回率 (Recall)，假阳性率 (FPR) 仅为 0.0018。
* **经济价值：** 在 1 亿元总持仓测试中，Plan B 成功规避 73.2 万元 ST 损失，整体净经济价值比无模型干预状态少亏损 145.7 万元，具备卓越的实盘应用潜力。

### 运行说明

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

---

## English Version

### Project Background & Business Pain Points
This project aims to proactively predict high-risk stocks in the A-share market by utilizing Large Language Models (LLM) to deeply analyze historical corporate announcements. The goal is to identify stocks that may face Special Treatment (ST) or delisting warnings by the exchange.

In quantitative trading and private equity, forecasting such extreme risks in advance holds core commercial value. When an ST event occurs, traditional quantitative models generally face devastating pain points:
1. **Liquidity Depletion & Data Distortion:** Extreme risks cause an immediate loss of market liquidity, rendering quantitative trading completely invalid and execution signals meaningless.
2. **Devastating Drawdown:** Accompanied by extreme delisting risks, stock net values suffer irreversible crashes, causing massive principal losses.
3. **Compliance & System Hard Blocks:** Upon triggering a high-risk status, risk management systems deploy hard blocks, preventing quantitative trading instructions from being dispatched entirely.

Furthermore, traditional prediction models relying on financial indicators or shallow text mining (non-LLM methods) tend to miss implicit risk warnings (such as hidden financial dangers, normative flaws, and illegal risks) and fail to capture signals that require complex logical reasoning and global contextual understanding. Therefore, this project leverages a multi-modal LLM to extract deep features from unstructured data, outputting a daily "blacklist" of high-risk stocks. This assists risk management systems in conducting proactive position adjustments before the formal event occurs, avoiding unnecessary forced liquidations and opportunity costs.

### Core Methodology & Model Architecture Design
Processing A-share announcement data from 2019 to 2024, the system achieves high-precision warnings through a multi-stage automated pipeline and a dual-track warning framework (Plan A and Plan B).

#### 1. LLM Structured Risk Parsing Module
The system utilizes the multi-modal LLM **Qwen3-8B** to perform inference on massive announcement texts, generating continuous risk scores (0-100) across four dimensions:
* **Financial Risk:** Identifies signals reflecting financial deterioration and delisting pressure, such as negative net profit, revenue below 300 million RMB, and negative audit opinions.
* **Normative Risk:** Identifies signals reflecting information disclosure and governance flaws, such as delayed annual reports, disclosure deficiencies, and capital occupation.
* **Illegal Risk:** Identifies signals reflecting major violations and regulatory penalties, such as financial fraud, major violations, and fraudulent issuance.
* **Other Risks:** Identifies signals reflecting operational anomalies and liquidity risks, such as frozen bank accounts, illegal guarantees, and ongoing operational anomalies.

#### 2. Plan A: Rule-Driven Risk Warning System
Plan A is a business logic system based on expert rules and grid search.
* **Risk Aggregation Design:** It adopts the **Maximum (Max)** value among the four-dimensional risk scores as the comprehensive risk score. The motivation for this design is that ST risks are typically triggered by a single extreme high-risk event, making extreme risk dimensions far more important than average risk levels.
* **Temporal Decay & Dual-Threshold Mechanism:** ST risks are persistent, and the impact of an announcement does not vanish overnight. The system implements a "temporal risk memory decay mechanism" combined with an Entry Threshold and a Retain Threshold. When there are no new high-risk announcements, the risk status decays over time until it falls below the Retain Threshold to be removed from the blacklist.
* **Parameter Optimization:** Grid Search is utilized to find optimal parameters across different cost budgets (e.g., maximum blacklist size of 100 or 150), achieving a balance between recall rate and blacklist scale.

#### 3. Plan B: Machine Learning Warning System (Advanced Version)
Building upon Plan A, Plan B introduces a supervised learning framework for secondary modeling.
* **Continuous Label System Design:** Traditional approaches use 0/1 binary classification (Normal/ST), ignoring the progressive nature of risk development. This project innovatively constructs labels as the inverse of the days remaining until the next ST event: `y_t = 1/(M-N)` (where M is the future ST start date and N is the current date). This allows the model to directly learn the evolutionary trajectory of risk urgency.
* **Multi-Dimensional Feature Engineering:**
  * *Temporal Decay:* The four original LLM scores decay independently daily (e.g., coefficient of 0.5), with the new score generated as `max(old_value, new_value)`.
  * *Horizontal Aggregation:* Extracts daily maximum risk and the count of non-zero dimensions to reflect the extremity and breadth of risks.
  * *Longitudinal Smoothing:* Calculates the monthly risk mean to smooth out single-day noise.
  * *Cyclical Encoding:* Uses `sin(2πt/12)` and `cos(2πt/12)` to capture time-periodic patterns such as earnings reporting seasons.
* **Modeling & Hyperparameter Tuning:** The model uses the `HistGradientBoostingRegressor`, which natively supports missing values and histogram acceleration. The Optuna Bayesian optimization algorithm is employed to automatically tune hyperparameters, utilizing `Precision - FPR` as the objective function.

### Evaluation System & Economic Value
The ST prediction task has a severely imbalanced characteristic (ST occurrences account for approximately 1.2% of the whole market). In the business logic, the cost of false negatives (an average principal loss exceeding 50%) is extremely asymmetrical, far outweighing the cost of false positives (an annualized opportunity cost of 5%). Therefore, the evaluation criteria dictate that **the priority of Recall is absolutely higher than Precision**.

In the 2024 test set backtest:
* **Performance:** Plan B achieved a high Recall of 65.3% and an exceptionally low False Positive Rate (FPR) of 0.0018 by maintaining an average daily blacklist of only 9.3 stocks (which is 1/12 the size of Plan A).
* **Economic Value:** In a simulated test with a total portfolio of 100 million RMB, Plan B successfully avoided 732,000 RMB in ST-related losses. Overall, it reduced losses by 1.457 million RMB compared to the no-model baseline, demonstrating outstanding practical application potential.

### Execution Instructions

# MY Capital LLM Prediction for ST

## PlanA

1. `main.py`
   Full LLM pipeline. It performs inference starting from announcement texts, generates risk scores, and outputs the blacklist.

2. `run_planA_with_4d_score.py`
   Directly reads the retained four-dimensional risk assessment scores. It generates the blacklist and evaluation results based on the max aggregation of the 4D scores.

## PlanB

1. `auto_tune.py`
   Parameter tuning module.

2. Model Training Module Launch Command
   The command below uses relative paths within the current project. By default, input files are placed in the `PlanB/` directory, and outputs are written to `PlanB/output/st_status_model/`.

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
