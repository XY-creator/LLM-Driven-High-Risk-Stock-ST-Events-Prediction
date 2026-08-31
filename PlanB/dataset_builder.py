# dataset_builder

"""
【模块作用】
本模块负责将 data_loader 输出的标准化公告数据加工为 LLM 可直接使用的样本，
即“按股票-日期组织后的模型输入数据”。

【主要职责】
1. 按股票与日期聚合公告，并按时间窗口筛选可见公告；
2. 进行文本构建与压缩（标题截断、全文关键句抽取、长度控制、条数限制）；
3. 基于风险关键词优先保留高风险信号内容；
4. 产出统一输入字段（如 stock_code、trade_date、ann_text）供 inference 调用。

【关键配置】
- MAX_ANN_PER_DAY
- USE_FULLTEXT_KEY_SENTENCES
- CONTENT_KEY_MAX_SENTENCES / CONTENT_KEY_MAX_CHARS / CONTENT_KEY_MIN_SENT_LEN
- TITLE_MAX_CHARS / SINGLE_ANN_MAX_CHARS
- RISK_KEYWORDS
- LOOKBACK_DAYS（若窗口逻辑在本模块实现）

【输出结果】
- 面向推理的样本表（每行一个“股票-日期”样本）
- 下游 llm_engine / inference 可直接消费的输入文本字段

"""

import re
import pandas as pd
import numpy as np
from collections import deque
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(it, **kwargs):
        return it

import config
import data_loader

def build_calendar(start_date, end_date):
    # 构建自然日日历
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    cal = pd.date_range(start, end, freq="D")
    return cal

def build_universe(ann_df, st_df):
    u1 = set(ann_df["stock_code"].unique())
    u2 = set(st_df["stock_code"].unique())
    universe = sorted(list(u1 | u2))
    return universe

def _clean_text(x):
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    s = str(x).strip()
    if s.lower() == "nan":
        return ""
    # 合并多空白
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    s = re.sub(r"\n{2,}", "\n", s)
    return s.strip()

def extract_key_sentences_from_full_content(
    full_content,
    keywords=None,
    max_sentences=3,
    max_chars=300,
    min_sent_len=8,
):
    keywords = keywords or config.RISK_KEYWORDS
    content = _clean_text(full_content)
    if not content:
        return ""

    if len(content) <= max_chars:
        return content

    raw_sents = re.split(r"[。！？；\n\r]+", content)
    raw_sents = [s.strip() for s in raw_sents if s and s.strip()]
    if not raw_sents:
        return content[:max_chars]

    picked = []
    seen = set()

    for s in raw_sents:
        if len(s) < min_sent_len:
            continue
        if s in seen:
            continue
        for kw in keywords:
            if kw and kw in s:
                picked.append(s)
                seen.add(s)
                break
        if len(picked) >= max_sentences:
            break

    if not picked:
        half = max_chars // 2
        head = content[:half].strip()
        tail = content[-half:].strip()
        out = head + "…" + tail if tail else head
        return out[:max_chars]

    out = "；".join(picked)
    return out[:max_chars]

def build_llm_text_input(ann_df, universe, calendar):
    step_bar = tqdm(total=7, desc="构建LLM输入", unit="step")

    ann_df = ann_df.copy()

    ann_df["publish_date"] = pd.to_datetime(ann_df["publish_date"], errors="coerce").dt.normalize()
    step_bar.update(1)

    cal_min = pd.Timestamp(calendar.min()).normalize()
    cal_max = pd.Timestamp(calendar.max()).normalize()
    ann_df = ann_df[(ann_df["publish_date"] >= cal_min) & (ann_df["publish_date"] <= cal_max)]
    step_bar.update(1)

    ann_df = ann_df.dropna(subset=["title", "category_name", "publish_date"])

    ann_df["title"] = ann_df["title"].map(_clean_text)
    ann_df["category_name"] = ann_df["category_name"].map(_clean_text)
    if "content" in ann_df.columns:
        ann_df["content"] = ann_df["content"].map(_clean_text)

    ann_df = ann_df[(ann_df["title"] != "") & (ann_df["category_name"] != "")]
    ann_df = ann_df.sort_values(by=["stock_code", "publish_date"])
    step_bar.update(1)

    ann_df["rank"] = ann_df.groupby(["stock_code", "publish_date"]).cumcount() + 1
    ann_df = ann_df[ann_df["rank"] <= int(config.MAX_ANN_PER_DAY)]

    if bool(getattr(config, "USE_FULLTEXT_KEY_SENTENCES", True)) and "content" in ann_df.columns:
        ann_df["content_key"] = ann_df["content"].apply(
            extract_key_sentences_from_full_content,
            keywords=config.RISK_KEYWORDS,
            max_sentences=int(config.CONTENT_KEY_MAX_SENTENCES),
            max_chars=int(config.CONTENT_KEY_MAX_CHARS),
            min_sent_len=int(config.CONTENT_KEY_MIN_SENT_LEN),
        )
    else:
        ann_df["content_key"] = ""
    step_bar.update(1)

    title_max = int(getattr(config, "TITLE_MAX_CHARS", 60))
    ann_max = int(getattr(config, "SINGLE_ANN_MAX_CHARS", 220))

    def _build_single(row):
        cat = row["category_name"]
        title = row["title"][:title_max]
        ck = row.get("content_key", "")
        if ck:
            s = f"{cat}：{title}；正文要点：{ck}"
        else:
            s = f"{cat}：{title}"
        return s[:ann_max]

    ann_df["single_ann"] = ann_df.apply(_build_single, axis=1)
    step_bar.update(1)

    full_index = pd.MultiIndex.from_product(
        [universe, calendar],
        names=["stock_code", "date"],
    )

    ann_df = ann_df.set_index(["stock_code", "publish_date"])
    ann_text = ann_df.groupby(level=["stock_code", "publish_date"])["single_ann"].apply("；".join)

    ann_text = ann_text.reindex(full_index, fill_value="")
    step_bar.update(1)

    ann_daily_df = ann_text.to_frame(name="ann_daily")
    ann_daily_df["ann_daily"] = ann_daily_df["ann_daily"].replace("", "无相关公告")

    ann_text_df = ann_text.to_frame(name="ann_text")
    win = int(config.LOOKBACK_DAYS)

    def _roll_concat_str(s):
        dq = deque()  
        out = []

        vals = s.to_numpy(dtype=object)
        for pos, v in enumerate(vals):
            if isinstance(v, str) and v:
                dq.append((pos, v))

            left = pos - win + 1
            while dq and dq[0][0] < left:
                dq.popleft()

            out.append("；".join(t for _, t in dq) if dq else "")

        return pd.Series(out, index=s.index)

    ann_text_df["ann_text"] = (
        ann_text_df.groupby(level="stock_code", sort=False, group_keys=False)["ann_text"]
        .apply(_roll_concat_str)
    )

    ann_text_df["ann_text"] = ann_text_df["ann_text"].replace("", "无相关公告")
    step_bar.update(1)
    step_bar.close()

    return ann_text_df, ann_daily_df

def build_daily_label_status(st_df, universe, calendar):
    st_df = st_df.copy()

    st_map = {}
    n_codes = st_df["stock_code"].nunique()
    for code, g in tqdm(
        st_df.groupby("stock_code", sort=False),
        total=n_codes,
        desc="构建ST区间映射",
        unit="stock",
    ):
        g = g.sort_values("entry_date")
        entries = g["entry_date"].to_numpy("datetime64[ns]")
        removes = g["remove_date"].to_numpy("datetime64[ns]")
        st_map[code] = (entries, removes)

    full_index = pd.MultiIndex.from_product([universe, calendar], names=["stock_code", "date"])
    y_status = pd.Series(0, index=full_index, dtype=np.int8)

    dates_np = calendar.to_numpy("datetime64[ns]")

    for code in tqdm(universe, desc="构建日度ST标签", unit="stock"):
        if code not in st_map:
            continue

        entries, removes = st_map[code]

        prev_i = np.searchsorted(entries, dates_np, side="right") - 1
        valid = prev_i >= 0  

        in_st = np.zeros(len(calendar), dtype=bool)
        in_st[valid] = dates_np[valid] <= removes[prev_i[valid]]

        y_status.loc[(code, slice(None))] = in_st.astype(np.int8)

    return pd.DataFrame({"y_status": y_status})

def build_daily_xy(start_date, end_date):
    print("  [1/6] 读取公告数据")
    ann_df = data_loader.load_raw_announcements()
    print("  [2/6] 清洗公告数据")
    ann_df = data_loader.clean_ann(ann_df)

    print("  [3/6] 读取ST数据")
    st_df = data_loader.load_raw_ST()
    print("  [4/6] 清洗ST数据")
    st_df = data_loader.clean_ST(st_df)

    print("  [5/6] 构建日历和股票池")
    calendar = build_calendar(start_date, end_date)
    universe = build_universe(ann_df, st_df)

    print("  [6/6] 构建LLM输入与标签")
    llm_input, ann_daily = build_llm_text_input(ann_df, universe, calendar)
    labels = build_daily_label_status(st_df, universe, calendar)

    data = pd.concat([llm_input, labels], axis=1)

    return data, calendar, universe, ann_daily