# config

from pathlib import Path

# ====================================================
# 0) 项目路径（全局）
# ====================================================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

# 相对项目目录的路径，避免依赖机器上的绝对路径。
RAW_DATA_DIR = PROJECT_DIR / "raw_data"
ST_HISTORY_DIR = PROJECT_DIR / "ST_history"
ST_HISTORY_FILE = "st_stock_history_2020_2023.csv"

OUTPUT_ROOT_DIR = BASE_DIR / "output"
DATA_DIR = OUTPUT_ROOT_DIR / "data"
PROCESSED_DATA_DIR = OUTPUT_ROOT_DIR / "processed"

ARTIFACT_DIR = OUTPUT_ROOT_DIR / "artifacts"
PREDICT_DIR = OUTPUT_ROOT_DIR / "output"


# ====================================================
# 1) 运行模式（data_loader）
# ====================================================
IS_PART_MODE = False
TEST_CONFIG = {
    "sample_size_per_year": 5000,
    #"sample_size_per_year": 200,
}

# ====================================================
# 2) 时间窗口 & 评估边界（main / inference / evaluator）
# ====================================================
OUTPUT_START_DATE = "2024-01-01"      # 系统输出黑名单开始日期
OUTPUT_END_DATE = "2024-12-31"

LOOKBACK_DAYS = 60                    # 每个评估日回看公告窗口（天）
HORIZON_DAYS = 365                    # 预测时间上限
MIN_LEAD_DAYS = 1                     # 预测时间下限

ONLY_USE_EVENTS_WITHIN_OUTPUT = True  # 严格评估开关（是否仅统计 output 内事件）
MARKET_N = 5400                       # 市场股票总量（用于 FPR 分母）

# ====================================================
# 3) 原始数据字段映射（data_loader）
# ====================================================
ANNOUNCEMENT_COLUMN_MAPPING = {
    "公告标识ID": "id",
    "公告日期": "publish_date",
    "标题": "title",
    "来源": "source",
    "是否有全文": "has_full_text",
    "全文内容": "content",
    "市场代码": "stock_code",
    "证券简称": "stock_name",
    "分类代码": "category_code",
    "分类名称": "category_name",
    "层级数": "level",
}

ANNOUNCEMENT_DTYPE = {
    "公告标识ID": str,
    "公告日期": str,
    "标题": str,
    "来源": str,
    "是否有全文": int,
    "全文内容": str,
    "市场代码": str,
    "证券简称": str,
    "分类代码": str,
    "分类名称": str,
    "层级数": str,
}

ST_COLUMN_MAPPING = {
    "s_info_windcode": "stock_code",
    "entry_dt": "entry_date",
    "remove_dt": "remove_date",
    "s_type_st": "st_type",
}

ST_DTYPE = {
    "s_info_windcode": str,
    "entry_dt": str,
    "remove_dt": float,
    "s_type_st": str,
}

# ====================================================
# 4) 文本构建与压缩（dataset_builder）
# ====================================================
MAX_ANN_PER_DAY = 3                   # 每天最多保留公告条数

USE_FULLTEXT_KEY_SENTENCES = True     # 是否从全文抽关键句
CONTENT_KEY_MAX_SENTENCES = 3         # 最多关键句数
CONTENT_KEY_MAX_CHARS = 300           # 关键句总长度上限
CONTENT_KEY_MIN_SENT_LEN = 8          # 最短句长过滤

TITLE_MAX_CHARS = 60                  # 标题截断
SINGLE_ANN_MAX_CHARS = 220            # 单条公告拼接后最大长度

RISK_KEYWORDS = [
    "ST", "*ST", "退市", "退市风险", "终止上市", "暂停上市", "恢复上市",
    "违法", "立案", "调查", "监管", "处罚", "违纪", "处分",
    "问询", "关注函", "监管函", "纪律", "监管", "停牌",
    "亏损", "净利润", "扣非为负", "营收下降", "收入下降",
    "减值", "商誉减值", "资产减值", "信用减值", "安全",
    "违约", "逾期", "欠款", "无法偿还", "流动性", "违法",
    "诉讼", "仲裁", "执行", "强制执行", "冻结", "查封",
    "担保", "违规", "占用", "整改",
    "审计", "意见", "保留意见", "否定意见", "强调", "经营", 
    "重整", "破产", "清算", "股份", "清偿",
    "造假", "遗漏", "营业额","风险","资产","资金",
    "财报", "信息披露", "股东", 
    "依法", "受理", "欺诈", "虚假", "瞒", "否定",
    "违规", "失信", "不达标", "不确定",
    "清算", "重整", "解散", "强制",
    "严重", "重大",
]


# ====================================================
# 5) LLM 模型与生成参数（llm_engine）
# ====================================================
LLM_MODEL_PATH = PROJECT_DIR / "models" / "Qwen3-8B"
LLM_LOAD_IN_4BIT = False

LLM_MAX_INPUT_TOKENS = 1024           # 输入 token 截断长度
LLM_ANN_TEXT_MAX_CHARS = 2500         # 进入 token 化前文本字符截断

LLM_MAX_NEW_TOKENS = 80               # 单次生成最大新 token（提速）
LLM_REPETITION_PENALTY = 1.05         # 重复惩罚


# ====================================================
# 6) 推理策略开关（inference）
# ====================================================
LLM_INFER_SCHEME = "fusion_decay"     # "scheme_a" / "fusion_decay"

LLM_INFER_ONLY_ON_ANN_DAYS = True     # True: 仅公告日推理
LLM_BOOTSTRAP_ON_OUTPUT_START = False # True: 输出首日全股票推理一次
LLM_SKIP_EMPTY_ANN_TEXT = True        # True: 空文本不送模型，记 0

LLM_SCORE_FFILL = True                # True: 分数前向填充
LLM_FFILL_MAX_DAYS = 30               # 超过天数不再沿用旧分（None=无限）

# baseline 融合策略（fusion_decay）参数
FUSION_PREFILTER_ENABLE = True
FUSION_FILTER_HIGH_TH = 8
FUSION_FILTER_MEDIUM_TH = 5
FUSION_DECAY_PER_DAY = 0.5
FUSION_ALLOWED_LABELS = ("HIGH", "MEDIUM")
FUSION_USE_DAILY_TEXT = True


# ====================================================
# 7) 并行与硬件（llm_engine）
# ====================================================
LLM_GPU_IDS = "auto"                  # "auto" / "0" / "0,1" / "0-3"
LLM_BATCH_SIZE = 4                    # 每卡 micro-batch


# ====================================================
# 8) 输出策略（signal_builder / main）
# ====================================================
RISK_SCORE_THRESHOLD = 80             # 黑名单阈值
BLACKLIST_USE_TOPK_CAP = False
BLACKLIST_TOPK = 100
SUBMIT_PREFIX = "blacklist"


# ====================================================
# 9) 调试与运行产物（llm_engine / utils）
# ====================================================
LLM_RUNS_DIR = ARTIFACT_DIR / "llm_runs"
LLM_DEBUG_LOG = True
LLM_DEBUG_TEXT_TRUNC = 1200
