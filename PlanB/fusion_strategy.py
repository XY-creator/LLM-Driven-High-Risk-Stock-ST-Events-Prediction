import re
import numpy as np
import pandas as pd

import config

class AdvancedFastFilter:
    """
    baseline 风格公告预筛:
    - HIGH: filter_score >= high_th
    - MEDIUM: filter_score >= medium_th
    - LOW: 其余
    """

    def __init__(self, keywords=None, category_terms=None, high_th=4, medium_th=2):
        self.keywords = keywords or list(getattr(config, "RISK_KEYWORDS", []))
        self.category_terms = category_terms or [
            "风险提示", "退市风险", "ST风险", "警示公告", "特别处理", "风险警示",
            "重大事项", "业绩预告", "业绩修正", "诉讼", "调查", "处罚", "监管", "异常波动",
        ]
        self.high_th = int(high_th)
        self.medium_th = int(medium_th)

        kw_escaped = [re.escape(str(k)) for k in self.keywords if str(k).strip()]
        cat_escaped = [re.escape(str(k)) for k in self.category_terms if str(k).strip()]

        self.risk_regex = re.compile("|".join(kw_escaped), re.IGNORECASE) if kw_escaped else None
        self.category_regex = re.compile("|".join(cat_escaped), re.IGNORECASE) if cat_escaped else None

    def classify_text(self, text):
        if text is None:
            text = ""
        text = str(text).strip()

        if text == "" or text == "无相关公告":
            return "LOW", 0

        text_slice = text[:1200]
        keyword_cnt = 0
        if self.risk_regex is not None:
            matches = self.risk_regex.findall(text_slice)
            keyword_cnt = len(set(matches)) if matches else 0
        keyword_score = min(keyword_cnt, 10)

        category_score = 0
        if self.category_regex is not None and self.category_regex.search(text_slice):
            category_score = 2

        content_len = max(len(text_slice), 1)
        risk_density = keyword_cnt / (content_len / 1000.0 + 1e-6)
        density_score = 2 if risk_density > 2.0 else 0

        score = int(keyword_score + category_score + density_score)
        if score >= self.high_th:
            return "HIGH", score
        if score >= self.medium_th:
            return "MEDIUM", score
        return "LOW", score

    def classify_series(self, ann_daily_s):
        if isinstance(ann_daily_s, pd.DataFrame):
            if "ann_daily" in ann_daily_s.columns:
                ann_daily_s = ann_daily_s["ann_daily"]
            elif ann_daily_s.shape[1] == 1:
                ann_daily_s = ann_daily_s.iloc[:, 0]
            else:
                raise ValueError("ann_daily DataFrame 必须包含 'ann_daily' 列或仅一列。")

        labels = []
        scores = []
        for v in ann_daily_s.astype(str).to_list():
            lb, sc = self.classify_text(v)
            labels.append(lb)
            scores.append(sc)

        out = pd.DataFrame(
            {
                "filter_label": pd.Series(labels, index=ann_daily_s.index, dtype=object),
                "filter_score": pd.Series(scores, index=ann_daily_s.index, dtype=np.int16),
            }
        )
        return out


def apply_decay_memory_scores(
    raw_score_df,
    output_mask,
    decay_per_day=0.5,
    threshold=0.0,
    memory_mask=None,
):
    """
    baseline 风格记忆衰减（四维度扩展版）:
    - 每日各维度独立衰减
    - 当天某维度有新分时，用 max(旧分, 新分) 更新
    - 未入记忆前，只有新分 >= threshold 才入记忆
    - 低于 threshold 的记忆直接清空为 0
    """
    decay = float(decay_per_day)
    thr = float(threshold)

    if not isinstance(raw_score_df, pd.DataFrame):
        raise ValueError("raw_score_df 必须是 pd.DataFrame。")

    out = raw_score_df.copy()
    out.loc[:] = np.nan

    if memory_mask is None:
        memory_mask = output_mask
    memory_mask = np.asarray(memory_mask, dtype=bool)
    output_mask = np.asarray(output_mask, dtype=bool)

    mem_idx = raw_score_df.index[memory_mask]
    if len(mem_idx) == 0:
        return out.fillna(0.0).astype(np.float32)

    out_raw = raw_score_df.loc[mem_idx].sort_index(level=["stock_code", "date"])
    cols = out_raw.columns

    def _fuse_one_stock(df):
        out_df = pd.DataFrame(index=df.index, columns=cols, dtype=np.float32)
        
        # 遍历 4 个维度，分别执行衰减
        for col in cols:
            state = 0.0
            arr = df[col].to_numpy(dtype=np.float32)
            vals = np.zeros(len(arr), dtype=np.float32)

            for i, v in enumerate(arr):
                state = max(0.0, state - decay)

                if np.isfinite(v):
                    v = float(v)
                    if v > 0:
                        if state > 0:
                            state = max(state, v)
                        elif v >= thr:
                            state = v

                if state < thr:
                    state = 0.0

                vals[i] = state
                
            out_df[col] = vals
            
        return out_df

    out_fused = out_raw.groupby(level="stock_code", sort=False).apply(_fuse_one_stock)
    if isinstance(out_fused.index, pd.MultiIndex) and out_fused.index.nlevels == 3:
        out_fused.index = out_fused.index.droplevel(0)

    out.loc[mem_idx] = out_fused.reindex(mem_idx).astype(np.float32)
    out.loc[~output_mask, cols] = 0.0
    out = out.fillna(0.0).astype(np.float32)
    return out