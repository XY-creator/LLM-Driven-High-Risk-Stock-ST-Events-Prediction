# -*- coding: utf-8 -*-
"""
功能：
1) 内置 ST 可解释性规则（不依赖其它项目模块）
2) 读取 blacklist_reason.json
3) 为每条样本追加 explain 字段
4) 输出 enriched JSON

"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union


RISK_TYPE_PRIORITY: Dict[str, int] = {
    "重大违法信披类": 1,
    "审计意见异常类": 2,
    "监管调查处罚类": 3,
    "持续经营破产类": 4,
    "财务异常类": 5,
    "债务诉讼流动性类": 6,
}

# 保持 6 类风险词典内容不变，只是把 bucket 名字简化为 strong / medium / weak
ST_RULES: Dict[str, Dict[str, List[str]]] = {
    "财务异常类": {
        "strong": [
            "净利润为负",
            "扣除非经常性损益后净利润为负",
            "扣非净利润为负",
            "营业收入低于",
            "净资产为负",
        ],
        "medium": ["商誉减值", "资产减值", "信用减值", "业绩亏损"],
        "weak": ["亏损", "减值", "营收下降", "收入下降", "下滑"],
    },
    "审计意见异常类": {
        "strong": [
            "无法表示意见",
            "否定意见",
            "保留意见",
            "带强调事项段",
            "强调事项段",
        ],
        "medium": ["非标准审计意见", "审计报告"],
        "weak": ["审计机构", "审计意见"],
    },
    "监管调查处罚类": {
        "strong": [
            "被立案调查",
            "立案调查",
            "行政处罚",
            "行政处罚决定书",
            "纪律处分",
            "自律监管",
            "监管函",
            "问询函",
            "关注函",
        ],
        "medium": ["证监会", "证监局", "交易所", "被调查"],
        "weak": ["立案", "调查", "监管措施", "处罚", "问询", "关注"],
    },
    "重大违法信披类": {
        "strong": [
            "重大违法",
            "财务造假",
            "信息披露违法",
            "虚假记载",
            "误导性陈述",
            "重大遗漏",
        ],
        "medium": ["信披违法", "造假"],
        "weak": ["违法", "虚假", "遗漏"],
    },
    "债务诉讼流动性类": {
        "strong": [
            "无法偿还",
            "流动性紧张",
            "强制执行",
            "申请强制执行",
        ],
        "medium": ["债务违约", "贷款逾期", "账户冻结", "资产查封"],
        "weak": ["违约", "逾期", "欠款", "诉讼", "仲裁", "冻结", "查封", "流动性"],
    },
    "持续经营破产类": {
        "strong": [
            "持续经营存在重大不确定性",
            "持续经营重大不确定性",
            "破产清算",
            "破产重整",
        ],
        "medium": ["重整", "破产", "清算"],
        "weak": ["持续经营", "资不抵债"],
    },
}

BUCKET_RANK: Dict[str, int] = {
    "strong": 0,
    "medium": 1,
    "weak": 2,
}

NEG_WORDS: List[str] = [
    "不涉及",
    "不存在",
    "未触及",
    "未被",
    "未发生",
    "未受到",
    "未收到",
    "无需",
    "不构成",
    "已消除",
    "风险可控",
    "保证公告内容真实",
    "保证信息披露",
]

DISCLAIMER_WHITELIST_PHRASES: List[str] = [
    "本公告不存在虚假记载、误导性陈述或重大遗漏",
    "没有虚假记载、误导性陈述或重大遗漏",
    "不存在虚假记载、误导性陈述或重大遗漏",
    "本公司及董事会全体成员保证信息披露内容的真实、准确、完整",
    "本公司及董事会全体成员保证信息披露内容真实、准确、完整",
    "保证公告内容真实、准确、完整",
    "保证信息披露内容的真实、准确、完整",
    "保证公告内容的真实、准确、完整",
    "及董事会全体成员保证信息披露内容的真实、准确、完整",
    "董事会全体成员保证信息披露内容的真实、准确、完整",
    "全体成员保证信息披露内容的真实、准确、完整",
    "保证信息披露内容真实、准确、完整",
    "不存在虚假记载",
    "不存在误导性陈述",
    "不存在重大遗漏",
    "没有虚假记载",
    "没有误导性陈述",
    "没有重大遗漏",
    "无虚假记载",
    "无误导性陈述",
    "无重大遗漏",
    "导性陈述或重大遗漏",
    "陈述或重大遗漏",
    "性陈述或重大遗漏",
]

RISK_LEVEL_RANK: Dict[str, int] = {"high": 3, "medium": 2, "low": 1}


def _merge_spans(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not spans:
        return []
    spans = sorted(spans)
    out: List[Tuple[int, int]] = [spans[0]]
    for s, e in spans[1:]:
        ps, pe = out[-1]
        if s <= pe:
            out[-1] = (ps, max(pe, e))
        else:
            out.append((s, e))
    return out


def _build_disclaimer_spans(text: str) -> List[Tuple[int, int]]:
    raw: List[Tuple[int, int]] = []
    for phrase in DISCLAIMER_WHITELIST_PHRASES:
        if not phrase:
            continue
        start = 0
        while True:
            pos = text.find(phrase, start)
            if pos < 0:
                break
            raw.append((pos, pos + len(phrase)))
            start = pos + 1
    return _merge_spans(raw)


def _is_fully_inside_disclaimer(pos: int, end: int, disclaimer_spans: List[Tuple[int, int]]) -> bool:
    for s, e in disclaimer_spans:
        if pos >= s and end <= e:
            return True
    return False


def _has_negation_at_position(text: str, pos: int, keyword: str) -> bool:
    if not text or not keyword:
        return False
    win_start = max(0, pos - 18)
    win_end = min(len(text), pos + len(keyword) + 18)
    window = text[win_start:win_end]
    for neg in NEG_WORDS:
        if neg in window:
            return True
    return False


def keyword_has_valid_hit(text: str, keyword: str, disclaimer_spans: List[Tuple[int, int]]) -> bool:
    if not keyword:
        return False
    start = 0
    while True:
        pos = text.find(keyword, start)
        if pos < 0:
            return False
        end = pos + len(keyword)
        if _is_fully_inside_disclaimer(pos, end, disclaimer_spans):
            start = pos + 1
            continue
        if not _has_negation_at_position(text, pos, keyword):
            return True
        start = pos + 1


def _aggregate_risk_level(buckets_present: List[str]) -> str:
    if "strong" in buckets_present:
        return "high"
    if "medium" in buckets_present:
        return "medium"
    return "low"


def _sort_matched_keywords(by_kw: Dict[str, str]) -> List[str]:
    return sorted(
        by_kw.keys(),
        key=lambda k: (BUCKET_RANK.get(by_kw[k], 99), -len(k)),
    )


def _pick_better_bucket(prev: str, new: str) -> str:
    return prev if BUCKET_RANK.get(prev, 99) <= BUCKET_RANK.get(new, 99) else new


def _collect_hits_for_category(
    text: str,
    rules: Dict[str, List[str]],
    disclaimer_spans: List[Tuple[int, int]],
) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    for bucket in ("strong", "medium", "weak"):
        for kw in rules.get(bucket, []):
            if not kw or kw not in text:
                continue
            if not keyword_has_valid_hit(text, kw, disclaimer_spans):
                continue
            hits.append({"keyword": kw, "bucket": bucket})
    return hits


def match_ann_daily_to_risk_types(text: str) -> List[Dict[str, Any]]:
    if not text or not str(text).strip():
        return []

    text = str(text).strip()
    disclaimer_spans = _build_disclaimer_spans(text)

    results: List[Dict[str, Any]] = []

    for risk_type, rules in ST_RULES.items():
        hit_list = _collect_hits_for_category(text, rules, disclaimer_spans)
        if not hit_list:
            continue

        by_kw: Dict[str, str] = {}
        for h in hit_list:
            kw = h["keyword"]
            bucket = h["bucket"]
            if kw not in by_kw:
                by_kw[kw] = bucket
            else:
                by_kw[kw] = _pick_better_bucket(by_kw[kw], bucket)

        matched_keywords = _sort_matched_keywords(by_kw)
        buckets_present = list({by_kw[k] for k in matched_keywords})
        risk_level = _aggregate_risk_level(buckets_present)

        results.append(
            {
                "risk_type": risk_type,
                "risk_level": risk_level,
                "matched_keywords": matched_keywords,
            }
        )

    results.sort(key=lambda r: RISK_TYPE_PRIORITY.get(r["risk_type"], 99))
    return results


def _derive_primary_secondary(
    risk_types: List[Dict[str, Any]],
) -> Tuple[Optional[str], List[str]]:
    if not risk_types:
        return None, []

    sorted_by_severity = sorted(
        risk_types,
        key=lambda r: (
            -RISK_LEVEL_RANK.get(r.get("risk_level", ""), 0),
            RISK_TYPE_PRIORITY.get(r["risk_type"], 99),
        ),
    )
    primary = sorted_by_severity[0]["risk_type"]
    rest = [r["risk_type"] for r in sorted_by_severity[1:]]
    rest_sorted = sorted(rest, key=lambda rt: RISK_TYPE_PRIORITY.get(rt, 99))
    return primary, rest_sorted


def _risk_level_label_zh(level: str) -> str:
    return {"high": "高", "medium": "中", "low": "低"}.get(level, level)


def _build_overall_explanation(
    risk_types: List[Dict[str, Any]],
    primary: Optional[str],
) -> str:
    if not risk_types or not primary:
        return "该公告未匹配到明确的 ST 相关风险短语，建议结合全文与其他指标综合判断。"

    primary_entry = next((r for r in risk_types if r["risk_type"] == primary), risk_types[0])
    pk = primary_entry.get("matched_keywords") or []
    pk_str = "、".join(pk[:8]) if pk else "相关表述"

    if len(risk_types) == 1:
        name = primary_entry["risk_type"]
        rl = primary_entry["risk_level"]
        lab = _risk_level_label_zh(rl)
        if rl == "high":
            return (
                f"该公告整体命中{name}{lab}风险信号，主要关键词包括{pk_str}，"
                f"表明公司存在较高风险，需重点关注。"
            )
        if rl == "medium":
            return (
                f"该公告整体命中{name}{lab}风险信号，主要关键词包括{pk_str}，"
                f"表明存在较明显的风险迹象，建议结合年报与审计意见等综合判断。"
            )
        return (
            f"该公告整体命中{name}{lab}风险提示，主要关键词包括{pk_str}，"
            f"提示一般性风险关注，可酌情跟踪。"
        )

    names_ordered = sorted(
        [r["risk_type"] for r in risk_types],
        key=lambda n: RISK_TYPE_PRIORITY.get(n, 99),
    )
    if len(names_ordered) == 2:
        type_part = f"{names_ordered[0]}与{names_ordered[1]}"
    else:
        type_part = "、".join(names_ordered[:-1]) + f"与{names_ordered[-1]}"

    pl = primary_entry["risk_level"]
    plab = _risk_level_label_zh(pl)
    levels = {r["risk_type"]: r["risk_level"] for r in risk_types}

    if len(set(levels.values())) > 1:
        return (
            f"该公告整体命中{type_part}等风险信号，其中「{primary}」风险等级更高（{plab}），"
            f"主要关键词包括{pk_str}等。"
        )
    return (
        f"该公告整体命中{type_part}等风险信号，均以{plab}等级为主，"
        f"主要关键词包括{pk_str}等，建议结合公告全文综合判断。"
    )


def explain_row(
    row_or_ann_daily: Union[str, Dict[str, Any]],
    risk_score: Optional[Union[float, int]] = None,
) -> Dict[str, Any]:
    if isinstance(row_or_ann_daily, dict):
        ann = row_or_ann_daily.get("ann_daily")
        ann = ann if isinstance(ann, str) else ""
    else:
        ann = row_or_ann_daily if isinstance(row_or_ann_daily, str) else ""

    _ = risk_score  # 保留参数，但不动 risk_score
    risk_types = match_ann_daily_to_risk_types(ann)
    primary, secondary = _derive_primary_secondary(risk_types)
    overall = _build_overall_explanation(risk_types, primary)

    return {
        "risk_types": risk_types,
        "primary_risk_type": primary,
        "secondary_risk_types": secondary,
        "overall_explanation": overall,
    }


def load_reason_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def enrich_with_explain(data: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for date_key, rows in data.items():
        if not isinstance(rows, list):
            out[date_key] = rows
            continue

        new_rows: List[Dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                new_rows.append(row)
                continue

            merged = dict(row)
            merged["explain"] = explain_row(row)
            new_rows.append(merged)

        out[date_key] = new_rows
    return out


def write_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="为 blacklist_reason.json 批量生成 ST explain 并导出 JSON（简化版）")
    p.add_argument("-i", "--input", required=True, help="输入 blacklist_reason.json 路径")
    p.add_argument("-o", "--output-json", required=True, help="输出带 explain 的 JSON 路径")
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    in_path = os.path.abspath(args.input)
    if not os.path.isfile(in_path):
        print(f"输入文件不存在: {in_path}", file=sys.stderr)
        return 1

    data = load_reason_json(in_path)
    enriched = enrich_with_explain(data)

    out_json = os.path.abspath(args.output_json)
    write_json(out_json, enriched)
    print(f"已写入 JSON: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())