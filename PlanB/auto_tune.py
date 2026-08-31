import optuna
import shutil
import sys
from pathlib import Path

# 导入现有模块
import st_status_model
from evaluator import evaluate_blacklist_json

BASE_DIR = Path(__file__).resolve().parent

# 保存原始的实例化函数，防止后续动态替换时引发无限递归
_orig_make_model = st_status_model._make_model
_orig_make_regressor = st_status_model._make_regressor

def objective(trial):
    # ==========================================
    # 1. 业务与采样超参数 (控制黑名单规模倾向)
    # ==========================================
    negative_ratio = trial.suggest_float("negative_ratio", 10.0, 40.0)
    zero_ratio = trial.suggest_float("zero_ratio", 2.0, 10.0)
    label_threshold = trial.suggest_float("label_threshold", 0.001, 0.05, log=True)
    
    # ==========================================
    # 2. 树模型底层超参数 (控制模型拟合能力)
    # ==========================================
    # 学习率越低，需要的 max_iter (树的数量) 通常越多，上限稍微调高到 250
    max_iter = trial.suggest_int("max_iter", 80, 250) 
    # 学习率对梯度提升树至关重要，采用对数刻度在 0.01 到 0.2 之间搜索
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
    # 控制单棵树的复杂度，值越小越不容易过拟合
    max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 15, 63)
    # L2 正则化项，加大可以压制模型对噪音特征的权重分配
    l2_regularization = trial.suggest_float("l2_regularization", 1e-4, 5.0, log=True)

    # 【动态注入 / Monkey Patch】
    # 动态覆写原模块内的函数，让它在实例化模型时带入 Optuna 提供的超参数
    def patched_make_model(random_state):
        model = _orig_make_model(random_state)
        model.set_params(
            learning_rate=learning_rate,
            max_leaf_nodes=max_leaf_nodes,
            l2_regularization=l2_regularization
        )
        return model

    def patched_make_regressor(random_state):
        reg = _orig_make_regressor(random_state)
        reg.set_params(
            learning_rate=learning_rate,
            max_leaf_nodes=max_leaf_nodes,
            l2_regularization=l2_regularization
        )
        return reg

    # 将补丁打到 st_status_model 模块上
    st_status_model._make_model = patched_make_model
    st_status_model._make_regressor = patched_make_regressor

    # ==========================================
    # 3. 运行模型训练与评估
    # ==========================================
    trial_out_dir = BASE_DIR / "output" / "auto_tune" / f"trial_{trial.number}"
    trial_out_dir.mkdir(parents=True, exist_ok=True)

    class Args: pass
    args = Args()
    args.risk_score_csv = str(BASE_DIR / "four_dimension_risk_score.csv")
    args.st_label_csv = str(BASE_DIR / "ST_history_label.csv")
    args.output_dir = str(trial_out_dir)
    args.train_start_date = "2020-01-01"
    args.train_end_date = "2022-12-31"
    args.predict_start_date = "2023-01-01"
    args.predict_end_date = "2023-12-31"
    args.validation_start = "2022-01-01"
    
    args.negative_ratio = negative_ratio
    args.zero_ratio = zero_ratio
    args.label_threshold = label_threshold
    args.max_iter = max_iter
    args.random_state = 10101
    
    args.history_daily_json_name = "daily_st_status_2020_2022.json"
    args.predict_2024_daily_json_name = "daily_st_status_2023_predicted.json"
    args.history_interval_json_name = "st_status_intervals_2020_2022.json"
    args.predict_2024_interval_json_name = "st_status_intervals_2023_predicted.json"
    args.blacklist_daily_json_name = "daily_blacklist_predicted.json"
    args.predict_2024_label_json_name = "st_label_2023_predicted.json"

    try:
        st_status_model.run(args)
    except Exception as e:
        # 捕捉因为极端超参数（如过大的学习率导致的爆炸）引发的错误并剪枝
        raise optuna.TrialPruned()

    blacklist_path = trial_out_dir / args.blacklist_daily_json_name
    try:
        metrics, _ = evaluate_blacklist_json(
            blacklist_json_path=str(blacklist_path),
            risk_score_csv=args.risk_score_csv,
            st_label_csv=args.st_label_csv,
            start_date="2023-01-01",
            end_date="2023-12-31",
            output_dir=str(trial_out_dir)
        )
    except ValueError as e:
        raise optuna.TrialPruned()

    acc = metrics.get("Acc", 0.0) if metrics.get("Acc") is not None else 0.0
    fpr = metrics.get("FPR", 1.0) if metrics.get("FPR") is not None else 1.0

    # 记录独立的 Acc 和 FPR 到该次 Trial 记录中
    trial.set_user_attr("Acc", acc)
    trial.set_user_attr("FPR", fpr)

    score = acc - fpr 

    # 验证完后立刻删掉临时文件释放空间
    shutil.rmtree(trial_out_dir, ignore_errors=True)

    return score

if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.INFO)
    study = optuna.create_study(direction="maximize", study_name="ST_Model_Full_Tuning")
    
    print("开始全局参数自动调优，预计进行 50 轮迭代...")
    print("💡 提示：你可以随时按 Ctrl+C 中断调参，程序会自动打印截至目前的最好结果。\n")

    try:
        study.optimize(objective, n_trials=50, n_jobs=1) 
    except KeyboardInterrupt:
        print("\n" + "!"*50)
        print("检测到手动中断 (KeyboardInterrupt)！已停止探索新的参数。")
        print("!"*50 + "\n")

    print("="*50)
    if len(study.trials) == 0 or all(t.state != optuna.trial.TrialState.COMPLETE for t in study.trials):
        print("目前还没有成功完成的有效迭代轮次。")
        sys.exit(0)

    # 获取最优结果
    trial = study.best_trial
    print(f"👑 最佳综合得分 (Acc - FPR): {trial.value:.4f}")
    
    best_acc = trial.user_attrs.get('Acc', 0.0)
    best_fpr = trial.user_attrs.get('FPR', 1.0)
    print(f"📊 对应的单一指标 -> Acc 准确率: {best_acc:.4f}  |  FPR 误报率: {best_fpr:.6f}")
    
    print("\n⚙️ 最佳超参数组合：")
    for key, value in trial.params.items():
        if isinstance(value, float):
            print(f"  --{key.replace('_', '-')} {value:.6f}")
        else:
            print(f"  --{key.replace('_', '-')} {value}")
