# llm_engine

import gc
import json
import multiprocessing as mp
import os
import re
import time
import warnings
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(it, **kwargs):
        return it

warnings.filterwarnings("ignore")

def parse_gpu_list(gpu_str):
    gpu_str = str(gpu_str).strip().lower()
    if gpu_str in ("", "none"):
        return []

    out = []
    for part in gpu_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(list(set(out)))

def resolve_gpu_ids(spec=None):
    spec = spec if spec is not None else getattr(config, "LLM_GPU_IDS", "auto")

    if not torch.cuda.is_available():
        return []

    if str(spec).strip().lower() == "auto":
        return list(range(torch.cuda.device_count()))

    gpu_ids = parse_gpu_list(str(spec))
    n = torch.cuda.device_count()
    gpu_ids = [i for i in gpu_ids if 0 <= i < n]
    return gpu_ids

def _truncate_keep_tail(text, max_chars):
    if text is None:
        return ""
    s = str(text)
    if len(s) <= max_chars:
        return s
    return "…(省略前文)…" + s[-max_chars:]

def build_llm_prompt(tokenizer, ann_text):
    ann_text = "" if ann_text is None else str(ann_text)
    ann_text = _truncate_keep_tail(ann_text, int(getattr(config, "LLM_ANN_TEXT_MAX_CHARS", 2500)))

    system_prompt = (
        "你将收到某只股票最近一段时间的公告信息（含标题与正文要点）。\n"
        "请根据上交所对于股票special treatment事件的四个角度风险定义，为我判断并输出这篇公告所呈现的四类风险的分别风险分数，每类风险给出 0-100 的整数分数（0=无风险，100=极高风险）。\n"
        "禁止输出思考过程、分析过程、解释说明、推理文字、前后缀、Markdown、代码块、标签。\n"
        "禁止输出 <think>、</think> 或任何类似内部思考标记。\n"
        "你的回复必须是且只能是一个完整 JSON 对象；如果不确定，也必须直接输出 JSON。\n"
        "回复必须以 { 开始，以 } 结束，且中间只能包含四个字段及对应整数值。\n"
        "字段名必须严格为 financial_risk、normative_risk、illegal_risk、other_risk，不允许增删字段。\n\n"
        "四类风险定义如下：\n\n"
        "1. 财务类强制退市风险（financial_risk）：\n"
        "   - 最近一个会计年度经审计的利润总额、净利润、扣非净利润孰低为负值，且营业收入低于3亿元（需扣除与主营业务无关的收入）；\n"
        "   - 最近一个会计年度期末净资产为负值；\n"
        "   - 财报被出具无法表示意见或否定意见的审计报告；\n"
        "   - 因追溯重述或行政处罚导致上述指标实际触及。\n\n"
        "2. 规范类强制退市风险（normative_risk）：\n"
        "   - 未在法定期限内披露年报/半年报，且停牌2个月后仍未披露；\n"
        "   - 财报存在重大会计差错或虚假记载，被责令改正但2个月内未改正；\n"
        "   - 半数以上董事无法保证财报真实，且2个月内未改正；\n"
        "   - 信息披露或规范运作存在重大缺陷，被要求改正但2个月内未改正；\n"
        "   - 控股股东及其关联人非经营性占用资金（余额≥净资产30%或≥2亿元），被责令改正后2个月内未改正；\n"
        "   - 连续2个会计年度财报内部控制被出具无法表示意见或否定意见；\n"
        "   - 股本总额或股权分布连续20个交易日不达标，停牌1个月内未解决；\n"
        "   - 公司可能被依法强制解散；\n"
        "   - 法院受理重整、和解或破产清算申请。\n\n"
        "3. 重大违法类强制退市风险（illegal_risk）：\n"
        "   - 欺诈发行（IPO或重组上市文件造假）；\n"
        "   - 重大信息披露违法（如年报虚假记载导致连续多年财务指标触及终止上市，或虚假记载金额巨大：一年≥2亿且占比≥30%；两年合计≥3亿且占比≥20%；连续三年存在虚假记载）；\n"
        "   - 危害国家安全、公共安全、生态安全、生产安全、公众健康等领域，情节恶劣，严重损害国家或社会公共利益。\n\n"
        "4. 其他风险警示（other_risk）：\n"
        "   - 控股股东及其关联人非经营性占用资金（≥净资产5%或>1000万元），1个月内未清偿；\n"
        "   - 违规对外担保（≥净资产5%或>1000万元），1个月内未整改；\n"
        "   - 董事会、股东会无法正常召开并形成有效决议；\n"
        "   - 财报内部控制被出具无法表示意见或否定意见；\n"
        "   - 生产经营活动受到严重影响且预计3个月内不能恢复正常；\n"
        "   - 主要银行账户被冻结；\n"
        "   - 连续3个会计年度扣非前后净利润孰低均为负，且最近一年审计报告显示持续经营能力不确定；\n"
        "   - 年报存在虚假记载但未触及重大违法退市；\n"
        "   - 分红不达标（主板：近三年累计分红＜年均净利润30%且＜5000万元）；\n"
        "   - 严重失信、持续经营能力重大不确定等其他情形。\n"
    )

    user_prompt = (
        f"【公告内容】\n{ann_text}\n\n"
        "【输出要求】\n"
        "1. 直接输出 JSON。\n"
        "2. 不要先写分析，不要输出 <think>。\n"
        "3. 不要写除 JSON 之外的任何字符。\n"
        "4. 必须以 { 开始，以 } 结束。\n"
        "5. 输出格式必须严格等于："
        '{"financial_risk": <0-100整数>, "normative_risk": <0-100整数>, "illegal_risk": <0-100整数>, "other_risk": <0-100整数>}'
    )

    chat_history = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        prompt = tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

def _strip_code_fences(s):
    if "```" not in s:
        return s.strip()
    parts = s.split("```")
    if len(parts) >= 3:
        return parts[1].strip()
    return s.strip()

def parse_risk_score(output_text):
    default_res = {
        "financial_risk": 0,
        "normative_risk": 0,
        "illegal_risk": 0,
        "other_risk": 0
    }
    
    if output_text is None:
        return default_res, False, None, "empty_output"

    text = _strip_code_fences(str(output_text)).strip()

    m = _JSON_RE.search(text)
    if m:
        js = m.group(0).strip()
        try:
            obj = json.loads(js)
            # 安全提取4个维度的分数，兜底为0并限制在0-100之间
            res = {
                "financial_risk": max(0, min(100, int(obj.get("financial_risk", 0)))),
                "normative_risk": max(0, min(100, int(obj.get("normative_risk", 0)))),
                "illegal_risk": max(0, min(100, int(obj.get("illegal_risk", 0)))),
                "other_risk":   max(0, min(100, int(obj.get("other_risk", 0)))),
            }
            return res, True, obj, None
        except Exception as e:
            return default_res, False, None, f"json_load_or_parse_failed: {type(e).__name__}: {e}"

    return default_res, False, None, "no_json_found"

def _select_torch_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16

def load_llm_on_device(device):
    tokenizer = AutoTokenizer.from_pretrained(
        config.LLM_MODEL_PATH,
        trust_remote_code=True,
        padding_side="left",
    )
    tokenizer.truncation_side = "left"

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }

    if bool(getattr(config, "LLM_LOAD_IN_4BIT", False)):
        try:
            from transformers import BitsAndBytesConfig
            qcfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=_select_torch_dtype(),
            )
            model_kwargs["quantization_config"] = qcfg
            model_kwargs["device_map"] = {"": device}
        except Exception as e:
            raise RuntimeError(
                "你开启了 LLM_LOAD_IN_4BIT=True，但当前环境无法加载 bitsandbytes 4bit。\n"
                f"原始错误：{type(e).__name__}: {e}\n"
                "解决：pip 安装 bitsandbytes（或改为 LLM_LOAD_IN_4BIT=False）。"
            )
    else:
        model_kwargs["torch_dtype"] = _select_torch_dtype()
        model_kwargs["device_map"] = {"": device}

    model = AutoModelForCausalLM.from_pretrained(
        config.LLM_MODEL_PATH,
        **model_kwargs,
    )
    model.eval()

    return model, tokenizer

def _infer_batches_on_worker(gpu_id, task_path, result_path, debug_path, summary_path, batch_size):
    t0 = time.time()

    if (gpu_id is not None) and (gpu_id >= 0) and torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = f"cuda:{gpu_id}"
    else:
        device = "cpu"

    model, tokenizer = load_llm_on_device(device)

    max_new_tokens = int(getattr(config, "LLM_MAX_NEW_TOKENS", 200))
    do_sample = False
    repetition_penalty = float(getattr(config, "LLM_REPETITION_PENALTY", 1.05))
    max_input_tokens = int(getattr(config, "LLM_MAX_INPUT_TOKENS", 1024))

    debug_on = bool(getattr(config, "LLM_DEBUG_LOG", True))
    debug_trunc = int(getattr(config, "LLM_DEBUG_TEXT_TRUNC", 1200))

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    n_total = 0
    n_ok = 0
    n_fail = 0

    rp = open(result_path, "w", encoding="utf-8")
    dp = open(debug_path, "w", encoding="utf-8") if debug_on else None

    batch_records = []
    batch_prompts = []

    def _flush_batch():
        nonlocal n_total, n_ok, n_fail, batch_records, batch_prompts

        if not batch_records:
            return

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_tokens,
        )
        if device.startswith("cuda"):
            inputs = {k: v.to(device) for k, v in inputs.items()}

        seq_len = int(inputs["input_ids"].shape[1])

        attn = inputs.get("attention_mask", None)
        if attn is None:
            in_lens = [seq_len] * int(inputs["input_ids"].shape[0])
        else:
            in_lens = attn.sum(dim=1).tolist()

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=None,         
            top_p=None,               
            top_k=None,               
            num_beams=1,              
            repetition_penalty=repetition_penalty,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=pad_token_id,
        )

        with torch.inference_mode():
            gen = model.generate(**inputs, **gen_kwargs)

        out_texts = tokenizer.batch_decode(gen[:, seq_len:], skip_special_tokens=True)

        for rec, out_text, in_len in zip(batch_records, out_texts, in_lens):
            n_total += 1

            out_text = (out_text or "").strip()
            rs_dict, ok, parsed_obj, err = parse_risk_score(out_text)

            if ok:
                n_ok += 1
            else:
                n_fail += 1

            # 写入四个维度的分数
            rp.write(
                json.dumps(
                    {
                        "stock_code": rec["stock_code"],
                        "date": rec["date"],
                        "financial_risk": rs_dict["financial_risk"],
                        "normative_risk": rs_dict["normative_risk"],
                        "illegal_risk": rs_dict["illegal_risk"],
                        "other_risk": rs_dict["other_risk"],
                        "parse_ok": bool(ok),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            if dp is not None:
                dp.write(
                    json.dumps(
                        {
                            "gpu_id": int(gpu_id),
                            "stock_code": rec["stock_code"],
                            "date": rec["date"],
                            "ann_text_tail": _truncate_keep_tail(rec.get("ann_text", ""), 600),
                            "prompt_chars": int(len(rec.get("_prompt", ""))),
                            "input_tokens": int(in_len),   
                            "seq_len": int(seq_len),       
                            "raw_output_trunc": _truncate_keep_tail(out_text, debug_trunc),
                            "parse_ok": bool(ok),
                            "parsed_obj": parsed_obj,
                            "risk_scores": rs_dict,
                            "error": err,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        del inputs, gen, out_texts
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        batch_records = []
        batch_prompts = []

    with open(task_path, "r", encoding="utf-8") as f:
        n_lines = sum(1 for _ in f)
        f.seek(0)
        for line in tqdm(f, total=n_lines, desc=f"GPU{gpu_id}推理", unit="sample"):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ann_text = rec.get("ann_text", "")

            prompt = build_llm_prompt(tokenizer, ann_text)

            rec["_prompt"] = prompt  
            batch_records.append(rec)
            batch_prompts.append(prompt)

            if len(batch_records) >= batch_size:
                _flush_batch()

                if (n_total % 50) == 0:
                    gc.collect()
                    if device.startswith("cuda"):
                        torch.cuda.empty_cache()

    _flush_batch()

    rp.close()
    if dp is not None:
        dp.close()

    summary = {
        "gpu_id": int(gpu_id),
        "device": device,
        "task_path": str(task_path),
        "result_path": str(result_path),
        "debug_path": str(debug_path) if debug_on else None,
        "n_total": int(n_total),
        "n_parse_ok": int(n_ok),
        "n_parse_fail": int(n_fail),
        "parse_ok_rate": float(n_ok / n_total) if n_total else 0.0,
        "elapsed_sec": float(time.time() - t0),
        "batch_size": int(batch_size),
        "max_input_tokens": int(max_input_tokens),
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(do_sample),
    }
    with open(summary_path, "w", encoding="utf-8") as sf:
        json.dump(summary, sf, ensure_ascii=False, indent=2)

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def parallel_infer_risk_scores(task_records, run_dir, gpu_ids=None, batch_size=None):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        mp.set_start_method("spawn", force=False)
    except RuntimeError:
        pass

    if gpu_ids is None:
        gpu_ids = resolve_gpu_ids()

    if batch_size is None:
        batch_size = int(getattr(config, "LLM_BATCH_SIZE", 4))

    if not gpu_ids:
        gpu_ids = [-1]

    tasks_dir = run_dir / "tasks"
    results_dir = run_dir / "results"
    debug_dir = run_dir / "debug"
    summary_dir = run_dir / "summary"
    for d in (tasks_dir, results_dir, debug_dir, summary_dir):
        d.mkdir(parents=True, exist_ok=True)

    n = len(task_records)
    if n == 0:
        meta = {
            "run_dir": str(run_dir),
            "n_tasks_total": 0,
            "n_results_total": 0,
            "gpu_ids": [],
            "batch_size": int(batch_size),
            "worker_specs": [],
            "summaries": [],
            "debug_dir": str(run_dir / "debug"),
            "results_dir": str(run_dir / "results"),
            "skipped_reason": "no_tasks",
        }
        return [], meta

    ng = len(gpu_ids)
    chunk = (n + ng - 1) // ng

    procs = []
    worker_specs = []

    for i, gid in enumerate(tqdm(gpu_ids, desc="切分并分发任务", unit="worker")):
        s = i * chunk
        e = min((i + 1) * chunk, n)
        part = task_records[s:e]
        task_path = tasks_dir / f"gpu_{gid}_tasks.jsonl"
        result_path = results_dir / f"gpu_{gid}_results.jsonl"
        debug_path = debug_dir / f"gpu_{gid}_debug.jsonl"
        summary_path = summary_dir / f"gpu_{gid}_summary.json"

        with open(task_path, "w", encoding="utf-8") as tf:
            for r in tqdm(part, desc=f"写入任务 gpu={gid}", unit="sample", leave=False):
                tf.write(json.dumps(r, ensure_ascii=False) + "\n")

        if gid == -1:
            _infer_batches_on_worker(
                gpu_id=-1,
                task_path=str(task_path),
                result_path=str(result_path),
                debug_path=str(debug_path),
                summary_path=str(summary_path),
                batch_size=batch_size,
            )
        else:
            p = mp.Process(
                target=_infer_batches_on_worker,
                args=(gid, str(task_path), str(result_path), str(debug_path), str(summary_path), batch_size),
            )
            p.start()
            procs.append(p)

        worker_specs.append(
            {
                "gpu_id": gid,
                "task_path": str(task_path),
                "result_path": str(result_path),
                "debug_path": str(debug_path),
                "summary_path": str(summary_path),
                "n_tasks": int(len(part)),
            }
        )

    for p in tqdm(procs, desc="等待GPU进程结束", unit="proc"):
        p.join()

    results = []
    for spec in tqdm(worker_specs, desc="汇总推理结果", unit="worker"):
        rp = Path(spec["result_path"])
        if not rp.exists():
            continue
        with open(rp, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"读取结果 gpu={spec['gpu_id']}", unit="line", leave=False):
                line = line.strip()
                if not line:
                    continue
                results.append(json.loads(line))

    summaries = []
    for spec in tqdm(worker_specs, desc="汇总worker摘要", unit="worker"):
        sp = Path(spec["summary_path"])
        if sp.exists():
            with open(sp, "r", encoding="utf-8") as f:
                summaries.append(json.load(f))

    meta = {
        "run_dir": str(run_dir),
        "n_tasks_total": int(n),
        "n_results_total": int(len(results)),
        "gpu_ids": gpu_ids,
        "batch_size": int(batch_size),
        "worker_specs": worker_specs,
        "summaries": summaries,
        "debug_dir": str(debug_dir),
        "results_dir": str(results_dir),
    }
    return results, meta