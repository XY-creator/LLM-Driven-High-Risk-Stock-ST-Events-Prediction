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
        "你是一个股票风险预测专家。\n"
        "你将收到某只股票最近一段时间的公告信息（含标题与正文要点）。\n"
        "请判断该股票在未来 1-365 天内发生 ST/*ST 风险的强弱，输出风险分数。\n"
        "严格要求：只输出一行、严格合法的 JSON，不要输出任何多余文本、解释、代码块。\n"
        'JSON 格式固定为：{"risk_score": <0-100 的整数>}\n'
    )

    user_prompt = (
        "公告信息如下（按时间序列拼接，越靠后越接近当前日期）：\n"
        f"{ann_text}\n\n"
        "请只输出 JSON："
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
_RS_RE = re.compile(r'"risk_score"\s*:\s*(-?\d+)')

def _strip_code_fences(s):
    if "```" not in s:
        return s.strip()
    parts = s.split("```")
    if len(parts) >= 3:
        return parts[1].strip()
    return s.strip()

def parse_risk_score(output_text):
    if output_text is None:
        return 0, False, None, "empty_output"

    text = _strip_code_fences(str(output_text)).strip()

    m = _JSON_RE.search(text)
    if m:
        js = m.group(0).strip()
        try:
            obj = json.loads(js)
            rs = int(obj.get("risk_score", 0))
            rs = max(0, min(100, rs))
            return rs, True, obj, None
        except Exception as e:
            err = f"json_load_failed: {type(e).__name__}: {e}"
    else:
        err = "no_json_found"

    m2 = _RS_RE.search(text)
    if m2:
        try:
            rs = int(m2.group(1))
            rs = max(0, min(100, rs))
            return rs, True, {"risk_score": rs}, None
        except Exception as e:
            return 0, False, None, f"regex_parse_failed: {type(e).__name__}: {e}"

    return 0, False, None, err

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
            temperature=None,         # <--- 显式设为 None，消除警告
            top_p=None,               # <--- 显式设为 None，消除警告
            top_k=None,               # <--- 显式设为 None，消除警告          
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
            rs, ok, parsed_obj, err = parse_risk_score(out_text)

            if ok:
                n_ok += 1
            else:
                n_fail += 1

            rp.write(
                json.dumps(
                    {
                        "stock_code": rec["stock_code"],
                        "date": rec["date"],
                        "risk_score": int(rs),
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
                            "risk_score": int(rs),
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