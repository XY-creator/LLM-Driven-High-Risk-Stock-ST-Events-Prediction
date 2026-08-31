import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import config
from fusion_strategy import apply_decay_memory_scores


# ========= 参数区 =========
CSV_PATH = config.BASE_DIR / "four_dimension_risk_score_old.csv"

ENTRY_THRESHOLD = 80
RETAIN_THRESHOLD = 65
DECAY_PER_DAY = 0.3

OUTPUT_START = "2022-01-01"
OUTPUT_END = "2023-12-31"

STOCK_CODE = None
# 如果想指定股票，改成：
# STOCK_CODE = "300071"


# ========= 读取四维风险分数 =========
df = pd.read_csv(CSV_PATH)

# 自动识别日期列
date_col = None
for c in ["date", "trade_date", "publish_date"]:
    if c in df.columns:
        date_col = c
        break

if date_col is None:
    raise ValueError("找不到日期列，请确认CSV里是否有 date / trade_date / publish_date")

if "stock_code" not in df.columns:
    raise ValueError("找不到 stock_code 列")

risk_cols = [
    "financial_risk",
    "normative_risk",
    "illegal_risk",
    "other_risk",
]

for c in risk_cols:
    if c not in df.columns:
        raise ValueError(f"找不到风险列: {c}")

df[date_col] = pd.to_datetime(df[date_col])
df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)

# 四维风险取最大值作为 Plan A 输入分数
df["raw_score"] = df[risk_cols].max(axis=1)

# 构造完整 stock-date 面板
df = df[["stock_code", date_col, "raw_score"]].rename(columns={date_col: "date"})
df = df.sort_values(["stock_code", "date"])

start = pd.Timestamp(OUTPUT_START)
end = pd.Timestamp(OUTPUT_END)
calendar = pd.date_range(start, end, freq="D")

all_codes = sorted(df["stock_code"].unique())
full_index = pd.MultiIndex.from_product(
    [all_codes, calendar],
    names=["stock_code", "date"]
)

raw_score = (
    df.set_index(["stock_code", "date"])["raw_score"]
    .reindex(full_index)
    .astype(np.float32)
)

# output mask：只在输出区间内计算
dates = raw_score.index.get_level_values("date")
output_mask = (dates >= start) & (dates <= end)

# ========= 调用现有双阈值衰减系统 =========
decay_score = apply_decay_memory_scores(
    raw_score_s=raw_score,
    output_mask=output_mask,
    decay_per_day=DECAY_PER_DAY,
    entry_threshold=ENTRY_THRESHOLD,
    retain_threshold=RETAIN_THRESHOLD,
)

# ========= 自动选一只“图比较好看”的股票 =========
if STOCK_CODE is None:
    tmp = decay_score.groupby(level="stock_code").max()
    candidates = tmp[tmp >= RETAIN_THRESHOLD]

    if len(candidates) == 0:
        STOCK_CODE = tmp.idxmax()
    else:
        STOCK_CODE = candidates.idxmax()

print(f"Selected stock: {STOCK_CODE}")

raw_one = raw_score.loc[STOCK_CODE].copy()
decay_one = decay_score.loc[STOCK_CODE].copy()

plot_df = pd.DataFrame({
    "raw_score": raw_one,
    "decay_score": decay_one,
})

plot_df = plot_df.loc[OUTPUT_START:OUTPUT_END]

# 只显示有风险波动的窗口，避免图太空
active_dates = plot_df.index[
    (plot_df["raw_score"].fillna(0) > 0) | (plot_df["decay_score"] > 0)
]

if len(active_dates) > 0:
    left = active_dates.min() - pd.Timedelta(days=30)
    right = active_dates.max() + pd.Timedelta(days=30)
    plot_df = plot_df.loc[left:right]

# ========= 画图 =========
plt.figure(figsize=(14, 6))

# 原始LLM分数：只画有分数的点
raw_points = plot_df[plot_df["raw_score"].notna()]
plt.scatter(
    raw_points.index,
    raw_points["raw_score"],
    label="原始四维max风险分数",
    alpha=0.7,
    s=28
)

# 衰减后的风险状态
plt.plot(
    plot_df.index,
    plot_df["decay_score"],
    label="双阈值衰减后的风险状态",
    linewidth=2.5
)

# 阈值线
plt.axhline(
    ENTRY_THRESHOLD,
    linestyle="--",
    linewidth=1.5,
    label=f"进入阈值 Entry={ENTRY_THRESHOLD}"
)

plt.axhline(
    RETAIN_THRESHOLD,
    linestyle=":",
    linewidth=1.8,
    label=f"退出阈值 Retain={RETAIN_THRESHOLD}"
)

# 黑名单区间阴影
in_blacklist = plot_df["decay_score"] >= RETAIN_THRESHOLD
plt.fill_between(
    plot_df.index,
    0,
    100,
    where=in_blacklist,
    alpha=0.12,
    label="黑名单区间"
)

plt.title(
    f"股票 {STOCK_CODE} 风险记忆衰减示意图\n"
    f"entry={ENTRY_THRESHOLD}, retain={RETAIN_THRESHOLD}, decay={DECAY_PER_DAY}",
    fontsize=15
)

plt.xlabel("日期")
plt.ylabel("风险分数")
plt.ylim(0, 105)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

out_path = f"decay_plot_{STOCK_CODE}_entry{ENTRY_THRESHOLD}_retain{RETAIN_THRESHOLD}_decay{DECAY_PER_DAY}.png"
plt.savefig(out_path, dpi=300)
plt.show()

print(f"图已保存到: {out_path}")
