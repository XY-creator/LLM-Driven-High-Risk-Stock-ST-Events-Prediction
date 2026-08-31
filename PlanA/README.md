.
├── main.py                         # Original LLM pipeline entry
├── run_planA_with_4d_score.py     # 4D-risk + PlanA pipeline entry
│
├── config.py
├── data_loader.py
├── dataset_builder.py
├── llm_engine.py
├── inference.py
├── fusion_strategy.py
├── signal_builder.py
├── evaluator.py
├── st_explainer.py
├── utils.py
│
└── results/


### Original LLM Pipeline
Original end-to-end LLM inference pipeline
Run: python main.py

### 4D Risk + PlanA Pipeline
Use pre-generated 4D risk scores for PlanA blacklist generation
Run：python run_planA_with_4d_score.py

### Outputs:
- blacklist json
- evaluation metrics
- daily statistics csv