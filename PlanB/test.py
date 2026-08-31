import pandas as pd
from pathlib import Path

def peek_non_zero_scores(file_path):
    print(f"正在读取文件: {file_path} ...\n")
    
    try:
        # 读取双索引 CSV 文件 (假设前两列为 stock_code 和 date)
        df = pd.read_csv(file_path, index_col=["stock_code", "date"])
    except FileNotFoundError:
        print(f"❌ 找不到文件: {file_path}\n请确认当前运行路径或文件是否已生成。")
        return

    # 明确我们的四个风险维度列
    score_cols = [
        "financial_risk", 
        "normative_risk", 
        "illegal_risk", 
        "other_risk"
    ]
    
    # 筛选逻辑：只要这四列中任意一列的值大于 0，就保留该行
    # 使用 .any(axis=1) 是最高效的行级布尔索引方式
    non_zero_df = df[(df[score_cols] > 0).any(axis=1)]
    
    # 提取前 20 行
    top_20_results = non_zero_df.head(20)
    
    total_non_zero = len(non_zero_df)
    
    print(f"===== 数据概览 =====")
    print(f"总记录数: {len(df)}")
    print(f"包含非零风险分数的记录数: {total_non_zero}")
    print(f"====================\n")
    
    if total_non_zero == 0:
        print("没有找到任何非零的风险分数记录。")
    else:
        print("📉 以下是头 20 个检出风险的股票及分数：\n")
        # 只打印出分数相关的列，保持终端输出清爽
        print(top_20_results[score_cols])

if __name__ == "__main__":
    # 根据我们上一步修复的路径，文件应该在 output 目录下
    # 如果你在根目录运行，路径应该是 "output/four_dimension_risk_score.csv"
    # 如果文件就在当前目录，直接改为 "four_dimension_risk_score.csv" 即可
    target_csv = "output/output/four_dimension_risk_score.csv" 
    peek_non_zero_scores(target_csv)