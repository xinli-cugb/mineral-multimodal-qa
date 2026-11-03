#!/usr/bin/env python3
"""
端到端评测 + LLM-as-Judge

从 test 划分按 query_type 分层抽 100 条问题，走完整链路：
    混合检索（Dense+BM25→RRF）→ Qwen3-Reranker-0.6B 重排 top-3
    → Qwen2.5-7B-Instruct 生成回答 → GPT-4 按 rubric 打分

评分维度（默认 rubric，1-5 分）：
    accuracy         准确性（与参考答案事实一致）
    completeness     完整性（覆盖问题所需关键信息）
    faithfulness     忠实度（回答有检索文档依据）
    non_hallucination 无幻觉（未出现无依据事实）
可选 --judge-mode binary 输出 correct / hallucinated 两个 0/1 判定，both 同时输出。

同时统计端到端检索命中：gold 是否在重排 top-3 / 检索 top-10。

用法（先冒烟 10 条再全量）：
    export OPENAI_API_KEY=sk-xxx
    python scripts/eval_judge.py --num-samples 10          # 默认 GPT-4 评分
    python scripts/eval_judge.py --num-samples 100
    python scripts/eval_judge.py --num-samples 10 --judge-backend local   # 本地 7B 自评

产物：
    /root/autodl-tmp/logs/eval/judge_{split}_{n}_{mode}_{judge}.json
    /root/autodl-tmp/logs/eval/judge_{split}_{n}_{mode}_{judge}.jsonl（逐条缓存，可断点续跑）
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from eval_common import (  # noqa: E402
    DEFAULT_EMBEDDER,
    DEFAULT_INDEX_DIR,
    DEFAULT_LLM,
    DEFAULT_OUT_DIR,
    DEFAULT_RERANKER,
    doc_id_map,
    load_split_queries,
    read_jsonl,
    save_json,
    stable_key,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("eval_judge")

JUDGE_SYSTEM = (
    "你是严谨的矿物学知识问答评测员。只依据给定的参考答案与检索文档判断回答质量，"
    "不要引入外部知识，也不要因为回答文风好而放宽事实标准。只输出一个 JSON 对象，"
    "不要输出解释性文字或 Markdown。"
)

RUBRIC_KEYS = ["accuracy", "completeness", "faithfulness", "non_hallucination"]
BINARY_KEYS = ["correct", "hallucinated"]


class OpenAIChatClient:
    """OpenAI 兼容接口的 Judge 客户端，保持与本地 llm_client 相同的 chat 调用方式"""

    def __init__(self, model: str, api_key: Optional[str] = None,
                 base_url: Optional[str] = None, retries: int = 3):
        from openai import OpenAI

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit(
                "❌ 缺少 OPENAI_API_KEY。请设置环境变量，或传 --judge-api-key；"
                "若想用本地模型评分请加 --judge-backend local"
            )
        self.model = model
        self.retries = retries
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url or os.getenv("OPENAI_BASE_URL") or None,
        )

    def chat(self, messages: List[Dict], temperature: float = 0.0,
             max_tokens: int = 256, **_) -> SimpleNamespace:
        last_error: Optional[Exception] = None
        for attempt in range(self.retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                content = resp.choices[0].message.content or ""
                return SimpleNamespace(content=content)
            except Exception as exc:  # 限流/网络错误重试
                last_error = exc
                time.sleep(2 ** attempt)
        raise RuntimeError(f"OpenAI judge 调用失败: {last_error}")


def parse_args():
    parser = argparse.ArgumentParser(description="端到端 + LLM-as-Judge 评测")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--judge-mode", default="rubric",
                        choices=["rubric", "binary", "both"])
    parser.add_argument("--judge-backend", default="openai", choices=["openai", "local"],
                        help="openai=GPT-4 接口评分；local=用本地 --llm-path 模型自评")
    parser.add_argument("--judge-model", default="gpt-4",
                        help="OpenAI 模型名，默认 gpt-4")
    parser.add_argument("--judge-api-key", default=None,
                        help="默认读环境变量 OPENAI_API_KEY")
    parser.add_argument("--judge-base-url", default=None,
                        help="OpenAI 兼容接口地址，默认读 OPENAI_BASE_URL")
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR,
                        help="仅用于按 gold id 取回参考答案文本")
    parser.add_argument("--embedder-path", type=Path, default=DEFAULT_EMBEDDER)
    parser.add_argument("--reranker-path", type=Path, default=DEFAULT_RERANKER)
    parser.add_argument("--llm-path", type=Path, default=DEFAULT_LLM)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--judge-max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="生成温度（judge 固定用接近贪心的 0.01）")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--resume", action="store_true", help="从逐条缓存续跑")
    return parser.parse_args()


# ==================== 分层抽样 ====================

def stratified_sample(items: List[Dict], num: int, seed: int) -> List[Dict]:
    """按 query_type 比例抽样，最大余数法分配名额，保证可复现"""
    if num >= len(items):
        return list(items)

    groups: Dict[str, List[Dict]] = {}
    for item in items:
        groups.setdefault(item.get("query_type", "unknown"), []).append(item)

    total = len(items)
    quotas: Dict[str, int] = {}
    remainders = []
    for qtype, rows in groups.items():
        exact = len(rows) / total * num
        base = min(int(exact), len(rows))
        quotas[qtype] = base
        remainders.append((exact - int(exact), qtype))

    remainders.sort(reverse=True)
    left = num - sum(quotas.values())
    while left > 0:
        progressed = False
        for _, qtype in remainders:
            if left <= 0:
                break
            if quotas[qtype] < len(groups[qtype]):
                quotas[qtype] += 1
                left -= 1
                progressed = True
        if not progressed:
            break

    rng = random.Random(seed)
    sampled = []
    for qtype, rows in groups.items():
        sampled.extend(rng.sample(rows, quotas[qtype]))
    rng.shuffle(sampled)
    return sampled


# ==================== Judge Prompt ====================

def _extract_json(text: str) -> Optional[Dict]:
    text = re.sub(r"```(?:json)?", "", text or "").strip().strip("`").strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def build_judge_messages(question: str, gold: str, contexts: List[str],
                         answer: str, mode: str) -> List[Dict]:
    ctx_block = "\n".join(
        f"[文档{idx}] {text[:500]}" for idx, text in enumerate(contexts, start=1)
    ) or "（无检索文档）"

    if mode == "rubric":
        task = (
            "请对【系统回答】从以下四个维度各打 1-5 分（整数，5 最好）：\n"
            "- accuracy：准确性，回答与参考答案事实一致的程度；\n"
            "- completeness：完整性，是否覆盖问题所需的关键信息；\n"
            "- faithfulness：忠实度，回答是否都能在检索文档中找到依据；\n"
            "- non_hallucination：无幻觉，是否出现参考答案与检索文档都没有的事实（5=完全没有）。\n"
            '输出 JSON：{"accuracy":x,"completeness":x,"faithfulness":x,'
            '"non_hallucination":x,"reason":"简要理由"}'
        )
    elif mode == "binary":
        task = (
            "请做二值判定：\n"
            "- correct：回答是否与参考答案事实一致（1 正确 / 0 错误）；\n"
            "- hallucinated：回答是否包含参考答案与检索文档都没有的事实（1 有 / 0 无）。\n"
            '输出 JSON：{"correct":x,"hallucinated":x,"reason":"简要理由"}'
        )
    else:
        task = (
            "请先打 1-5 分（整数，5 最好）：accuracy 准确性、completeness 完整性、"
            "faithfulness 忠实度、non_hallucination 无幻觉；"
            "再做二值判定：correct（1 正确 / 0 错误）、hallucinated（1 有 / 0 无）。\n"
            '输出 JSON：{"accuracy":x,"completeness":x,"faithfulness":x,'
            '"non_hallucination":x,"correct":x,"hallucinated":x,"reason":"简要理由"}'
        )

    user = (
        f"【问题】\n{question}\n\n"
        f"【参考答案】\n{gold[:600] or '（无）'}\n\n"
        f"【检索文档】\n{ctx_block}\n\n"
        f"【系统回答】\n{answer[:1500] or '（空）'}\n\n"
        f"【评分任务】\n{task}"
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]


def normalize_judgement(obj: Optional[Dict], mode: str) -> Optional[Dict]:
    """校验并裁剪 judge 输出，缺失或越界返回 None"""
    if not obj:
        return None
    keys = []
    if mode in ("rubric", "both"):
        keys += RUBRIC_KEYS
    if mode in ("binary", "both"):
        keys += BINARY_KEYS
    out = {}
    for key in keys:
        if key not in obj:
            return None
        try:
            value = int(round(float(obj[key])))
        except (TypeError, ValueError):
            return None
        if key in RUBRIC_KEYS:
            value = min(5, max(1, value))
        else:
            value = 1 if value else 0
        out[key] = value
    out["reason"] = str(obj.get("reason", ""))[:200]
    return out


# ==================== 汇总 ====================

def summarize(details: List[Dict], mode: str) -> Dict:
    ok = [d for d in details if d.get("judge")]
    summary: Dict = {
        "num_samples": len(details),
        "num_judged": len(ok),
        "judge_parse_failures": len(details) - len(ok),
    }

    hits3 = [d for d in details if d.get("retrieval_hit@3")]
    hits10 = [d for d in details if d.get("retrieval_hit@10")]
    summary["retrieval_hit@3"] = round(len(hits3) / len(details), 4) if details else 0.0
    summary["retrieval_hit@10"] = round(len(hits10) / len(details), 4) if details else 0.0
    summary["avg_total_time_s"] = round(
        sum(d.get("total_time", 0.0) for d in details) / len(details), 2
    ) if details else 0.0
    summary["avg_generation_time_s"] = round(
        sum(d.get("generation_time", 0.0) for d in details) / len(details), 2
    ) if details else 0.0

    if mode in ("rubric", "both") and ok:
        rubric = {}
        for key in RUBRIC_KEYS:
            values = [d["judge"][key] for d in ok]
            rubric[key] = {
                "mean": round(sum(values) / len(values), 3),
                "distribution": {str(s): values.count(s) for s in range(1, 6)},
            }
        rubric["overall_mean"] = round(
            sum(v for d in ok for v in (d["judge"][k] for k in RUBRIC_KEYS))
            / (len(ok) * len(RUBRIC_KEYS)), 3
        )
        summary["rubric"] = rubric

    if mode in ("binary", "both") and ok:
        correct = sum(d["judge"]["correct"] for d in ok)
        halluc = sum(d["judge"]["hallucinated"] for d in ok)
        summary["binary"] = {
            "correct_rate": round(correct / len(ok), 4),
            "hallucination_rate": round(halluc / len(ok), 4),
        }

    # 按问题类型看 retrieval 命中与通过率，便于定位薄弱题型
    by_type: Dict[str, List[Dict]] = {}
    for d in details:
        by_type.setdefault(d.get("query_type", "unknown"), []).append(d)
    summary["by_type"] = {
        qtype: {
            "num_samples": len(rows),
            "retrieval_hit@3": round(
                sum(1 for r in rows if r.get("retrieval_hit@3")) / len(rows), 4
            ),
            **({
                "correct_rate": round(
                    sum(r["judge"]["correct"] for r in rows if r.get("judge"))
                    / max(1, sum(1 for r in rows if r.get("judge"))), 4
                )
            } if mode in ("binary", "both") else {}),
        }
        for qtype, rows in sorted(by_type.items())
    }
    return summary


# ==================== 主流程 ====================

def main():
    args = parse_args()
    items = load_split_queries(args.split)
    samples = stratified_sample(items, args.num_samples, args.seed)
    print(f"📊 split={args.split}, 抽样 {len(samples)} 条（共 {len(items)} 条）")
    from collections import Counter
    print("   类型分布:", dict(Counter(x["query_type"] for x in samples)))

    docs = doc_id_map(args.index_dir)

    # 先生成 Judge 客户端，缺少 key 时尽早失败，不白加载 7B
    if args.judge_backend == "openai":
        judge_client = OpenAIChatClient(
            model=args.judge_model,
            api_key=args.judge_api_key,
            base_url=args.judge_base_url,
        )
        print(f"⚖️ Judge: {args.judge_model}（OpenAI 兼容接口）")
    else:
        judge_client = None
        print(f"⚖️ Judge: 本地 {args.llm_path}（同模型自评）")

    from pipeline.mineral_qa_pipeline import MineralQAPipeline

    print("🚀 初始化 Pipeline（视觉关闭、查询融合关闭）...")
    pipeline = MineralQAPipeline(
        embedder_model_path=str(args.embedder_path),
        reranker_model_path=str(args.reranker_path),
        llm_model_path=str(args.llm_path),
        device=args.device,
        top_k_retrieval=10,
        top_k_rerank=3,
        enable_visual=False,
        enable_rerank=True,
        enable_bm25=True,
        enable_query_fusion=False,
    )
    pipeline.initialize()
    if judge_client is None:
        judge_client = pipeline.llm_client

    mode_tag = args.judge_mode
    judge_tag = re.sub(r"[^0-9A-Za-z._-]+", "_", f"{args.judge_backend}-{args.judge_model}")
    base = args.out_dir / f"judge_{args.split}_{len(samples)}_{mode_tag}_{judge_tag}"
    cache_path = base.with_suffix(".jsonl")

    cached: Dict[str, Dict] = {}
    if args.resume and cache_path.exists():
        for row in read_jsonl(cache_path):
            cached[row["key"]] = row
        print(f"♻️ 已加载逐条缓存 {len(cached)} 条: {cache_path}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # 非续跑模式覆盖写，避免旧结果重复追加
    cache_fp = open(cache_path, "a" if args.resume else "w", encoding="utf-8")

    details: List[Dict] = []
    t_start = time.time()
    for i, item in enumerate(samples, start=1):
        question = item["query"]
        gold_id = int(item["gold_id"])
        gold_text = docs.get(gold_id, {}).get("text", "")
        key = stable_key(args.split, question)

        if key in cached:
            detail = cached[key]
            details.append(detail)
            print(f"   [{i}/{len(samples)}] 复用缓存: {question[:30]}")
            continue

        t0 = time.time()
        try:
            response = pipeline.run(
                text_query=question,
                return_details=True,
                temperature=args.temperature,
                max_tokens=args.max_new_tokens,
            )
        except Exception as exc:  # 单条失败不中断整体评测
            print(f"   [{i}/{len(samples)}] ❌ 生成失败: {exc}")
            detail = {
                "key": key,
                "query": question,
                "gold_id": gold_id,
                "query_type": item["query_type"],
                "gold": gold_text,
                "answer": "",
                "error": str(exc),
                "judge": None,
            }
            details.append(detail)
            cache_fp.write(json.dumps(detail, ensure_ascii=False) + "\n")
            cache_fp.flush()
            continue

        reranked = response.reranked_docs or []
        retrieved = response.retrieved_docs or []
        reranked_ids = [int(d["id"]) for d in reranked if d.get("id") is not None]
        retrieved_ids = [int(d["id"]) for d in retrieved if d.get("id") is not None]
        contexts = [d.get("text", "") for d in reranked[:3]]

        judge_raw, judge_obj = "", None
        try:
            judge_messages = build_judge_messages(
                question, gold_text, contexts, response.answer, args.judge_mode
            )
            judge_resp = judge_client.chat(
                judge_messages,
                temperature=args.judge_temperature,
                max_tokens=args.judge_max_tokens,
                repetition_penalty=1.0,
            )
            judge_raw = judge_resp.content if hasattr(judge_resp, "content") else str(judge_resp)
            judge_obj = normalize_judgement(_extract_json(judge_raw), args.judge_mode)
        except Exception as exc:
            print(f"   [{i}/{len(samples)}] ⚠️ Judge 失败: {exc}")

        detail = {
            "key": key,
            "query": question,
            "gold_id": gold_id,
            "query_type": item["query_type"],
            "gold": gold_text,
            "answer": response.answer,
            "retrieved_ids": retrieved_ids,
            "reranked_ids": reranked_ids,
            "retrieval_hit@3": gold_id in reranked_ids,
            "retrieval_hit@10": gold_id in retrieved_ids,
            "judge": judge_obj,
            "judge_raw": judge_raw,
            "generation_time": response.generation_time,
            "total_time": response.total_time,
        }
        details.append(detail)
        cache_fp.write(json.dumps(detail, ensure_ascii=False) + "\n")
        cache_fp.flush()

        status = "✅" if judge_obj else "⚠️"
        print(f"   [{i}/{len(samples)}] {status} {question[:30]} | "
              f"hit@3={detail['retrieval_hit@3']} | "
              f"judge={judge_obj if judge_obj else 'parse-fail'}")

    cache_fp.close()

    summary = summarize(details, args.judge_mode)
    payload = {
        "meta": {
            "split": args.split,
            "num_samples": len(samples),
            "seed": args.seed,
            "judge_mode": args.judge_mode,
            "judge_backend": args.judge_backend,
            "judge_model": args.judge_model if args.judge_backend == "openai" else str(args.llm_path),
            "embedder_path": str(args.embedder_path),
            "reranker_path": str(args.reranker_path),
            "llm_path": str(args.llm_path),
            "top_k_retrieval": 10,
            "top_k_rerank": 3,
            "elapsed_s": round(time.time() - t_start, 1),
        },
        "summary": summary,
        "details": details,
    }
    save_json(base.with_suffix(".json"), payload)

    print("\n=== 汇总 ===")
    print(f"样本数: {summary['num_samples']}, 成功评分: {summary['num_judged']}, "
          f"解析失败: {summary['judge_parse_failures']}")
    print(f"检索命中@3: {summary['retrieval_hit@3']}, "
          f"检索命中@10: {summary['retrieval_hit@10']}")
    if "rubric" in summary:
        for key in RUBRIC_KEYS:
            row = summary["rubric"][key]
            print(f"{key}: mean={row['mean']} 分布={row['distribution']}")
        print(f"四项总均分: {summary['rubric']['overall_mean']}")
    if "binary" in summary:
        print(f"正确率: {summary['binary']['correct_rate']}, "
              f"幻觉率: {summary['binary']['hallucination_rate']}")
    print(f"平均生成耗时: {summary['avg_generation_time_s']}s, "
          f"平均端到端耗时: {summary['avg_total_time_s']}s")


if __name__ == "__main__":
    main()
