# data_loader
"""
【模块作用】
本模块是项目的数据入口层，负责把原始 ann/st 数据读取并标准化为下游可直接使用的统一格式。

【主要职责】
1. 从 data/raw 读取公告数据与 ST 数据；
2. 按 config 中的 COLUMN_MAPPING / DTYPE 做列名映射和类型约束；
3. 做基础清洗（日期解析、缺失处理、格式统一）；
4. 支持 IS_PART_MODE 下的调试采样；
5. 输出标准化 DataFrame 给 dataset_builder / inference / evaluator 使用。

【输入依赖】
- 配置项：路径、字段映射、dtype、采样参数（来自 config.py）
- 原始文件：公告表（ann）、ST 事件表（st）

【输出结果】
- 标准化公告数据（announcements_df）
- 标准化 ST 事件数据（st_df）

【不负责】
- 文本构建与压缩（dataset_builder）
- LLM 推理（llm_engine / inference）
- 黑名单生成与评估（signal_builder / evaluator）
"""

import pandas as pd
import numpy as np
from pathlib import Path
import config
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(it, **kwargs):
        return it

def load_raw_announcements():
    all_dfs = [] # 用于暂时存储每年的 dataframe

    # 动态计算需要的年份，避免加载无关的早期数据
    start_dt = pd.Timestamp(config.OUTPUT_START_DATE) - pd.Timedelta(days=config.LOOKBACK_DAYS)
    end_dt = pd.Timestamp(config.OUTPUT_END_DATE)
    target_years = range(start_dt.year, end_dt.year + 1)

    # 1. 获取需要读取的列名 
    use_cols = list(config.ANNOUNCEMENT_COLUMN_MAPPING.keys())
    print(f"   模式: {'部分模式' if config.IS_PART_MODE else '全量模式'}")
    print(f"   输出年份: {list(target_years)}")

    # 2. 循环读取每年的数据
    for year in tqdm(target_years, desc="读取公告年文件", unit="year"):
        # 拼接文件路径
        file_path = config.RAW_DATA_DIR / 'announcements' / f"announcements_{year}.csv"
     
        nrows_arg = None
        if config.IS_PART_MODE:  # 如果是部分模式，设置 nrows 参数；否则为 None (读取所有)
            nrows_arg = config.TEST_CONFIG["sample_size_per_year"]    

        # 读取 CSV
        df_year = pd.read_csv(
            file_path, 
            usecols=use_cols,  # 只读需要的列
            dtype=config.ANNOUNCEMENT_DTYPE,
            low_memory=False,   # 防止混合类型警告
            nrows=nrows_arg
        )
        
        # 3. 重命名列 
        df_year.rename(columns=config.ANNOUNCEMENT_COLUMN_MAPPING, inplace=True)
        
        # 4. 添加到列表
        all_dfs.append(df_year)

    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df['publish_date'] = pd.to_datetime(final_df['publish_date'], format='mixed', errors='coerce')
    final_df = final_df.sort_values(by='publish_date', ignore_index=True)

    return final_df

def clean_ann(df):
    # 简化处理直接删除缺少stock_code的股票
    # 直接返回删除缺失值后的新DataFrame
    return df.dropna(subset=['stock_code'])

def load_raw_ST():
    # 1. 数据路径（支持多个候选目录兜底）
    st_file = str(getattr(config, "ST_HISTORY_FILE", "st_stock_history_2020_2023.csv"))
    cand_dirs = [
        Path(getattr(config, "ST_HISTORY_DIR", config.RAW_DATA_DIR / "ST_history")),
        Path(getattr(config, "RAW_DATA_DIR", Path("."))) / "ST_history",
    ]

    st_path = None
    for d in cand_dirs:
        p = d / st_file
        if p.exists():
            st_path = p
            break

    if st_path is None:
        checked = [str(d / st_file) for d in cand_dirs]
        raise FileNotFoundError(
            "未找到 ST 历史文件。请确认以下任一路径存在：\n- "
            + "\n- ".join(checked)
        )

    # 2. 读取数据
    use_cols = list(config.ST_COLUMN_MAPPING.keys())

    st_df = pd.read_csv(
            st_path,
            usecols=use_cols,
            dtype=config.ST_DTYPE,
            low_memory=False
        )

    # 3. 重命名列 
    st_df.rename(columns=config.ST_COLUMN_MAPPING, inplace=True)

    # 4. 数据清洗：日期格式转换
    # 戴帽日期
    st_df['entry_date'] = pd.to_datetime(st_df['entry_date'], format='%Y%m%d', errors='coerce')
    
    # 摘帽日期 (如果股票还没摘帽，这里会是空值，转换后为 NaT)
    st_df['remove_date'] = pd.to_datetime(st_df['remove_date'], format='%Y%m%d', errors='coerce')

    # 5. 数据清洗：股票代码标准化
    # 确保代码没有多余空格，统一格式
    st_df['stock_code'] = st_df['stock_code'].astype(str).str.strip()
    st_df['stock_code'] = st_df['stock_code'].str.split('.').str[0]

    return st_df

def clean_ST(df):
    # 1. 预处理
    st_df = df.copy()
    st_df['remove_date'] = st_df['remove_date'].fillna(pd.Timestamp('2099-12-31'))
    st_df = st_df.sort_values(by=['stock_code', 'entry_date'])  #按股票代码和进入日期排序

    cleaned_rows = []
    # 2. 核心逻辑
    grouped = st_df.groupby('stock_code')
    for stock_code, group in tqdm(
        grouped,
        total=st_df['stock_code'].nunique(),
        desc="清洗ST周期",
        unit="stock",
    ):
        current_cycle_end = pd.Timestamp.min    # 初始化当前周期的 截止时间

        for index, row in tqdm(
            group.iterrows(),
            total=len(group),
            desc=f"处理ST记录 {stock_code}",
            unit="row",
            leave=False,
        ):
            entry = row['entry_date']
            remove = row['remove_date']

            # 第一种情况：这是该股票的第一条记录，或者
            # 当前记录的进入时间 晚于 上一个周期的结束时间 (说明是摘帽后复发)
            if entry > current_cycle_end:
                # 记录这个新周期
                cleaned_rows.append(row.to_dict())    # 把这一行存入结果
                current_cycle_end = remove  # 更新当前周期的结束点

            # 第二种情况：当前记录的进入时间 在 上一个周期内 (说明是ST变*ST，或重复记录)
            else:
                # 忽略这条记录的 entry_dt，因为它不是“正常->异常”的转折点
                # 如果这条记录延长了摘帽时间（比如升级导致摘帽推迟），我们需要更新周期的结束时间
                if remove > current_cycle_end:
                    current_cycle_end = remove
                    cleaned_rows[-1]['remove_date'] = remove  #找到刚刚存进去的最后一条记录（也就是当前这个周期的代表），更新它的 remove_dt
                   
    cleaned_df = pd.DataFrame(cleaned_rows) # 重组 DataFrame
    return cleaned_df
