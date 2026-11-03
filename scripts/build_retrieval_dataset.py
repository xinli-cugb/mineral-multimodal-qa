#!/usr/bin/env python3
"""
构建矿物检索训练数据集（Qwen3-Embedding / Qwen3-Reranker 通用）

数据源: text_data/all_v1.json (JSONL, 6408 条权威结构化矿物记录)
产物:
  corpus.jsonl                       全量语料（重建 FAISS 时也用这个）
  embedding/{train,val,test}.json    {query, positive, negatives}，适配现有 multi_neg 训练脚本
  reranker/{train,val,test}.json     {query, response, rejected_response}，适配现有 list 训练脚本
  reranker/{train,val,test}_pairs.jsonl  {query, doc, label}，适配 Qwen3-Reranker 官方 yes/no 训练
  stats.json                         数据统计

设计要点:
  1. 正样本文档只用权威结构化字段（brief_characterization / purpose / origin），不使用 LLM 改写描述，避免幻觉进入训练集。
  2. 查询由结构化字段生成，只有文本里确实存在该属性时才生成，避免“问题无答案”的假正样本。
  3. 难负样本用 Qwen3-Embedding 语义召回 + 属性/名称隔离，排除同一矿物、共生组分、近重复文档，降低假负样本。
  4. 按矿物名称分组切分 train/val/test，避免同一种矿物的查询泄漏到不同集合。
"""
import argparse
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

INVALID_VALUES = {"", "不详", "未知", "无", "none", "n/a", "nan"}
MIX_SPLIT_RE = re.compile(r"[+\-/,，、]")
PAREN_RE = re.compile(r"[（(][^）)]*[）)]")

DEF_TEMPLATES = [
    "{name}是什么？",
    "介绍一下{name}。",
    "{name}是一种什么矿物？",
]
USAGE_TEMPLATES = [
    "{name}有什么用途？",
    "{name}主要用于什么？",
    "{name}可以用来做什么？",
]
ORIGIN_TEMPLATES = [
    "{name}产自哪里？",
    "{name}的产地是哪里？",
]
IDENT_TEMPLATES = [
    "如何鉴别{name}？",
    "{name}有哪些鉴别特征？",
]


def clean_text(value) -> str:
    """去 HTML 标签、压缩空白"""
    if value is None:
        return ""
    text = re.sub(r"<[^>]+>", "", str(value))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_valid(value) -> bool:
    return clean_text(value).lower() not in INVALID_VALUES


def load_records(path: str) -> List[Dict]:
    """兼容 JSONL / JSON 数组"""
    raw = Path(path).read_text(encoding="utf-8").strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return [data]
        return data
    except json.JSONDecodeError:
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no} JSON 解析失败: {exc}") from exc
        return records


def normalize_name(name: str) -> str:
    """去掉括号补充说明，作为切分/负样本隔离的基本单位"""
    text = PAREN_RE.sub("", clean_text(name))
    text = re.sub(r"\s+", "", text)
    return text or clean_text(name)


def split_components(name: str, english_name: str = "") -> Set[str]:
    """拆出共生/混合矿物的组分，用于负样本隔离"""
    comps: Set[str] = set()
    for source in (normalize_name(name), clean_text(english_name)):
        if not source:
            continue
        for part in MIX_SPLIT_RE.split(source):
            part = part.strip().lower()
            if part:
                comps.add(part)
    if not comps:
        comps.add(normalize_name(name).lower())
    return comps


def canon_value(value: Optional[str]) -> Optional[str]:
    """把 6-6.5 / 6～6.5 / 6至6.5 统一成 6~6.5"""
    if not value:
        return None
    value = re.sub(r"\s+", "", str(value))
    value = value.replace("～", "~").replace("—", "~").replace("至", "~")
    value = value.replace("-", "~")
    value = value.strip("~")
    return value or None


def first_group(pattern: str, text: str) -> Optional[str]:
    match = re.search(pattern, text)
    if not match:
        return None
    value = match.group(1).strip()
    return value or None


def parse_attributes(brief: str) -> Dict[str, str]:
    """从权威 brief_characterization 中抽取可出题的属性"""
    brief = clean_text(brief)
    attrs: Dict[str, str] = {}

    number = r"([0-9]+(?:\.[0-9]+)?(?:\s*(?:-|~|～|至)\s*[0-9]+(?:\.[0-9]+)?)?)"

    hardness = first_group(r"(?:莫氏硬度|摩氏硬度|硬度|H\s*=)[^0-9]{0,12}" + number, brief)
    if hardness:
        attrs["hardness"] = canon_value(hardness)

    density = first_group(
        r"(?:相对密度|比重|密度|S\.G\.\s*=)[^0-9]{0,12}" + number, brief
    )
    if density:
        attrs["density"] = canon_value(density)

    formula = first_group(
        r"(?:化学式(?:为|是|：|:)?|化学成分(?:为|是|：|:)?|成份(?:为|是|：|:)?)"
        r"[^A-Za-z0-9]{0,4}([A-Za-z0-9\u2080-\u209f\u00b2\u00b3\u2070-\u2079()（）·.,]+)",
        brief,
    )
    if formula:
        formula = formula.strip(".,，。；;、 ")
        if len(formula) >= 2 and re.search(r"[A-Za-z]", formula):
            attrs["formula"] = formula

    crystal = first_group(r"(等轴|三方|四方|六方|单斜|三斜|斜方|正交|菱面体)晶系", brief)
    if crystal:
        attrs["crystal"] = crystal + "晶系"

    luster = first_group(r"([\u4e00-\u9fa5]{1,4}光泽)", brief)
    if luster:
        attrs["luster"] = luster

    streak = first_group(
        r"条痕\s*(?:为|是|呈|：|:)?\s*([\u4e00-\u9fa5]{1,8}?)(?:，|,|。|；|;|$|\s)",
        brief,
    )
    if streak:
        streak = re.sub(r"^(?:为|是|呈)", "", streak).strip()
        if streak:
            attrs["streak"] = streak

    cleavage = first_group(r"(极完全|完全|不完全|中等|清楚|无)解理", brief)
    if cleavage:
        attrs["cleavage"] = cleavage

    transparency = first_group(r"(半透明|不透明|微透明|透明)", brief)
    if transparency:
        attrs["transparency"] = transparency

    color = first_group(
        r"(?:颜色|色泽|色调|标本颜色)(?:为|是|呈|：|:)?\s*([\u4e00-\u9fa5]{1,10}色)",
        brief,
    )
    if not color:
        color = first_group(r"([\u4e00-\u9fa5]{1,8}色)", brief)
    if color:
        color = re.sub(r"^(?:颜色|色泽|色调|标本颜色)", "", color).strip()
        if color:
            attrs["color"] = color

    special = [kw for kw in ("滑感", "磁性", "荧光", "放射性", "变彩", "性脆", "导电") if kw in brief]
    if special:
        attrs["special"] = "、".join(special)

    return attrs


def build_document(record: Dict, doc_id: int) -> Dict:
    name = clean_text(record.get("name"))
    english_name = clean_text(record.get("english_name"))
    brief = clean_text(record.get("brief_characterization"))
    if not is_valid(brief):
        brief = ""
    main_purpose = clean_text(record.get("main_purpose"))
    if not is_valid(main_purpose):
        main_purpose = ""
    specific_purpose = clean_text(record.get("specific_purpose"))
    if not is_valid(specific_purpose):
        specific_purpose = ""

    origin = ""
    for key in ("place_of_origin", "place_of_province", "place_of_country"):
        candidate = clean_text(record.get(key))
        if is_valid(candidate):
            origin = candidate
            break

    parts = [f"{name}（{english_name}）" if english_name else name]
    if brief:
        parts.append(f"特征：{brief}")
    if specific_purpose:
        parts.append(f"用途：{specific_purpose}")
    if main_purpose:
        parts.append(f"用途类别：{main_purpose}")
    if origin:
        parts.append(f"产地：{origin}")
    text = "；".join(parts)

    normalized = normalize_name(name)
    return {
        "id": doc_id,
        "name": name,
        "english_name": english_name,
        "normalized_name": normalized,
        "components": sorted(split_components(name, english_name)),
        "text": text,
        "brief": brief,
        "main_purpose": main_purpose,
        "specific_purpose": specific_purpose,
        "origin": origin,
        "attrs": parse_attributes(brief),
        "group": normalized,
        "brief_key": brief if len(brief) >= 20 else "",
    }


def build_english_counts(docs: Sequence[Dict]) -> Counter:
    counts = Counter()
    for doc in docs:
        if doc["english_name"]:
            counts[doc["english_name"].lower()] += 1
    return counts


def build_attribute_only_queries(docs: Sequence[Dict]) -> Dict[int, List[Tuple[str, str, List[str]]]]:
    """构造不含矿物名的“属性反查”查询，仅在语料内答案唯一时生成"""
    extra: Dict[int, List[Tuple[str, str, List[str]]]] = defaultdict(list)
    combo_specs = [
        (("formula",), "化学式为{formula}的矿物是什么？", ["formula"]),
        (
            ("hardness", "color"),
            "硬度为{hardness}、颜色为{color}的矿物是什么？",
            ["hardness", "color"],
        ),
        (
            ("streak", "hardness"),
            "条痕为{streak}、硬度为{hardness}的矿物是什么？",
            ["streak", "hardness"],
        ),
        (
            ("luster", "color"),
            "具有{luster}且颜色为{color}的矿物是什么？",
            ["luster", "color"],
        ),
        (
            ("transparency", "color"),
            "颜色为{color}、{transparency}的矿物是什么？",
            ["transparency", "color"],
        ),
        (
            ("density", "color"),
            "密度为{density}、颜色为{color}的矿物是什么？",
            ["density", "color"],
        ),
        (
            ("cleavage", "hardness"),
            "具有{cleavage}解理、硬度为{hardness}的矿物是什么？",
            ["cleavage", "hardness"],
        ),
        (
            ("crystal", "color"),
            "属于{crystal}、颜色为{color}的矿物是什么？",
            ["crystal", "color"],
        ),
    ]

    for keys, template, terms in combo_specs:
        groups: Dict[Tuple[str, ...], List[int]] = defaultdict(list)
        for doc in docs:
            values = [doc["attrs"].get(key) for key in keys]
            if all(values):
                groups[tuple(values)].append(doc["id"])
        for values, ids in groups.items():
            if len(ids) != 1:
                continue
            kwargs = {key: value for key, value in zip(keys, values)}
            query = template.format(**kwargs)
            extra[ids[0]].append((query, "attribute_only", [str(v) for v in values]))
    return extra


def pick(rng: random.Random, items: List[str], k: int) -> List[str]:
    if not items:
        return []
    k = min(k, len(items))
    return rng.sample(items, k)


def generate_queries(
    doc: Dict,
    rng: random.Random,
    english_unique: bool,
    attr_only: List[Tuple[str, str, List[str]]],
    max_queries: int,
) -> List[Dict]:
    """按查询类型分桶采样，保证覆盖不同问法"""
    name = doc["name"]
    english = doc["english_name"]
    attrs = doc["attrs"]
    buckets: Dict[str, List[Tuple[str, List[str]]]] = defaultdict(list)

    for template in DEF_TEMPLATES:
        buckets["definition"].append((template.format(name=name), []))
    for template in USAGE_TEMPLATES:
        if doc["specific_purpose"]:
            buckets["usage"].append((template.format(name=name), []))
    for template in ORIGIN_TEMPLATES:
        if doc["origin"]:
            buckets["origin"].append((template.format(name=name), []))
    for template in IDENT_TEMPLATES:
        if doc["brief"]:
            buckets["identification"].append((template.format(name=name), []))

    attribute_templates = {
        "hardness": ["{name}的硬度是多少？", "{name}的莫氏硬度是多少？"],
        "density": ["{name}的密度是多少？", "{name}的相对密度是多少？"],
        "color": ["{name}是什么颜色的？", "{name}有哪些颜色？"],
        "luster": ["{name}的光泽是什么？", "{name}具有什么光泽？"],
        "streak": ["{name}的条痕是什么颜色？"],
        "cleavage": ["{name}的解理如何？"],
        "transparency": ["{name}是透明的吗？", "{name}的透明度如何？"],
        "crystal": ["{name}属于什么晶系？"],
        "formula": ["{name}的化学成分是什么？", "{name}的化学式是什么？"],
    }
    for attr, templates in attribute_templates.items():
        if attr in attrs:
            for template in templates:
                buckets["attribute"].append((template.format(name=name), [attrs[attr]]))

    if english_unique and english:
        buckets["english"].append((f"What is {english}?", []))
        buckets["english"].append((f"What are the properties of {english}?", []))
        if "formula" in attrs:
            buckets["english"].append((f"What is the chemical formula of {english}?", [attrs["formula"]]))
        if doc["specific_purpose"]:
            buckets["english"].append((f"What is {english} used for?", []))

    for query, _, terms in attr_only:
        buckets["attribute_only"].append((query, terms))

    # 分桶配额：定义/用途/产地各 1，属性 2~3，反查 1，英文 1
    selected: List[Tuple[str, List[str], str]] = []
    quota = {
        "definition": 1,
        "usage": 1 if doc["specific_purpose"] else 0,
        "origin": 1 if doc["origin"] else 0,
        "identification": 1 if doc["brief"] and rng.random() < 0.35 else 0,
        "attribute": 3,
        "attribute_only": 1 if attr_only else 0,
        "english": 1 if english_unique and english else 0,
    }
    for bucket_name, limit in quota.items():
        if limit <= 0 or not buckets[bucket_name]:
            continue
        chosen = pick(rng, buckets[bucket_name], limit)
        for query, terms in chosen:
            selected.append((query, terms, bucket_name))

    # 不足时从属性/定义/用途桶补足
    if len(selected) < max_queries:
        used = {item[0] for item in selected}
        fill_order = ["attribute", "definition", "usage", "identification", "origin", "attribute_only", "english"]
        for bucket_name in fill_order:
            for query, terms in buckets[bucket_name]:
                if len(selected) >= max_queries:
                    break
                if query in used:
                    continue
                selected.append((query, terms, bucket_name))
                used.add(query)

    # 去重并截断
    deduped: List[Tuple[str, List[str], str]] = []
    seen = set()
    for query, terms, bucket_name in selected:
        if query in seen:
            continue
        seen.add(query)
        deduped.append((query, terms, bucket_name))

    return [
        {"query": query, "query_type": bucket_name, "terms": terms}
        for query, terms, bucket_name in deduped[:max_queries]
    ]


class Qwen3EmbeddingMiner:
    """只用于难负样本挖掘，使用官方 last-token pooling + query instruction"""

    QUERY_INSTRUCTION = (
        "Instruct: Given a mineral-related query, retrieve relevant mineral descriptions "
        "that answer the query\nQuery: "
    )

    def __init__(self, model_path: str, device: str = "cuda", batch_size: int = 64):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA 不可用，拒绝回退到 CPU。请检查驱动/OMP_NUM_THREADS，"
                    "或显式传入 --device cpu。"
                )
            torch.cuda.set_device(0)
            print(f"   使用 GPU: {torch.cuda.get_device_name(0)}", flush=True)
        self.device = device
        self.batch_size = batch_size

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        self.tokenizer = tokenizer

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=dtype
        ).to(self.device)
        self.model.eval()

    def encode(
        self,
        texts: Sequence[str],
        is_query: bool = False,
        max_length: int = 256,
    ) -> np.ndarray:
        torch = self.torch
        vectors: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = list(texts[start:start + self.batch_size])
                if is_query:
                    batch = [self.QUERY_INSTRUCTION + text for text in batch]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                outputs = self.model(**encoded)
                hidden = outputs.last_hidden_state
                lengths = encoded["attention_mask"].sum(dim=1) - 1
                batch_vectors = hidden[torch.arange(hidden.size(0), device=self.device), lengths]
                batch_vectors = torch.nn.functional.normalize(batch_vectors.float(), p=2, dim=1)
                vectors.append(batch_vectors.cpu().numpy())
        if not vectors:
            return np.zeros((0, 1), dtype=np.float32)
        return np.concatenate(vectors, axis=0)


def is_excluded(positive: Dict, candidate: Dict, terms: List[str]) -> bool:
    """负样本隔离：同矿物/共生组分/名称包含/近重复/同属性命中"""
    if candidate["id"] == positive["id"]:
        return True
    if candidate["group"] == positive["group"]:
        return True
    pos_name, cand_name = positive["normalized_name"], candidate["normalized_name"]
    if pos_name and cand_name and (pos_name in cand_name or cand_name in pos_name):
        return True
    if set(positive["components"]) & set(candidate["components"]):
        return True
    if positive["brief_key"] and candidate["brief_key"] == positive["brief_key"]:
        return True
    if terms:
        candidate_brief = candidate["brief"]
        if candidate_brief and all(term in candidate_brief for term in terms):
            return True
    return False


def mine_negatives(
    docs: Sequence[Dict],
    samples: Sequence[Dict],
    query_vectors: Optional[np.ndarray],
    doc_vectors: Optional[np.ndarray],
    num_negatives: int,
    hard_ratio: float,
    seed: int,
) -> List[List[int]]:
    """为每条样本选负样本；有向量时按语义相似度选难负样本，否则随机"""
    rng = random.Random(seed)
    results: List[List[int]] = []
    n_docs = len(docs)
    hard_n = min(num_negatives, max(1, int(round(num_negatives * hard_ratio))))

    if query_vectors is None or doc_vectors is None:
        for sample in samples:
            positive = docs[sample["positive_id"]]
            terms = sample["terms"]
            candidates = [
                doc["id"]
                for doc in docs
                if not is_excluded(positive, doc, terms)
            ]
            rng.shuffle(candidates)
            results.append(candidates[:num_negatives])
        return results

    chunk_size = 2048
    for start in range(0, len(samples), chunk_size):
        chunk = samples[start:start + chunk_size]
        sims = query_vectors[start:start + chunk_size] @ doc_vectors.T  # (b, N)
        for row, sample in enumerate(chunk):
            positive = docs[sample["positive_id"]]
            terms = sample["terms"]
            scores = sims[row]
            order = np.argsort(-scores)
            valid: List[int] = []
            for idx in order:
                idx = int(idx)
                if idx >= n_docs:
                    continue
                if is_excluded(positive, docs[idx], terms):
                    continue
                valid.append(idx)
                if len(valid) >= 300:
                    break

            if not valid:
                results.append([])
                continue

            hard_pool = valid[:30]
            easy_pool = valid[30:]
            n_hard = min(hard_n, len(hard_pool))
            n_easy = num_negatives - n_hard
            chosen = rng.sample(hard_pool, n_hard) if hard_pool else []
            if n_easy > 0:
                if len(easy_pool) >= n_easy:
                    chosen.extend(rng.sample(easy_pool, n_easy))
                else:
                    chosen.extend(easy_pool)
                    remaining = num_negatives - len(chosen)
                    pool = [idx for idx in valid if idx not in chosen]
                    chosen.extend(pool[:remaining])
            results.append(chosen[:num_negatives])
        print(f"  难负样本挖掘: {min(start + chunk_size, len(samples))}/{len(samples)}", flush=True)
    return results


def assign_splits(docs: Sequence[Dict], val_ratio: float, test_ratio: float, seed: int) -> Dict[int, str]:
    """按矿物名称分组切分，避免同矿物跨集合"""
    groups = sorted({doc["group"] for doc in docs})
    rng = random.Random(seed)
    rng.shuffle(groups)
    n_val = int(len(groups) * val_ratio)
    n_test = int(len(groups) * test_ratio)
    val_groups = set(groups[:n_val])
    test_groups = set(groups[n_val:n_val + n_test])
    split_of: Dict[int, str] = {}
    for doc in docs:
        if doc["group"] in val_groups:
            split_of[doc["id"]] = "val"
        elif doc["group"] in test_groups:
            split_of[doc["id"]] = "test"
        else:
            split_of[doc["id"]] = "train"
    return split_of


def validate(
    docs: Sequence[Dict],
    samples: Sequence[Dict],
    negatives: Sequence[List[int]],
    split_of: Dict[int, str],
) -> List[str]:
    errors: List[str] = []
    seen_queries: Dict[str, int] = {}
    for sample, neg_ids in zip(samples, negatives):
        query = sample["query"]
        positive_id = sample["positive_id"]
        if query in seen_queries and seen_queries[query] != positive_id:
            errors.append(f"查询重复且正样本不同: {query}")
        seen_queries[query] = positive_id
        if positive_id in neg_ids:
            errors.append(f"正样本出现在负样本中: {query}")
        if len(neg_ids) != len(set(neg_ids)):
            errors.append(f"负样本重复: {query}")
        for neg_id in neg_ids:
            if docs[neg_id]["text"] == docs[positive_id]["text"]:
                errors.append(f"负样本文本与正样本相同: {query}")
        if len(errors) > 20:
            break

    # 分组切分不重叠
    group_splits: Dict[str, Set[str]] = defaultdict(set)
    for doc in docs:
        group_splits[doc["group"]].add(split_of[doc["id"]])
    for group, splits in group_splits.items():
        if len(splits) > 1:
            errors.append(f"分组跨集合: {group} -> {splits}")
    return errors


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def write_jsonl(path: Path, rows: Iterable[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="构建矿物检索训练数据集")
    parser.add_argument("--input", default=str(PROJECT_ROOT / "text_data/all_v1.json"), help="权威结构化数据")
    parser.add_argument("--output_dir", default=str(PROJECT_ROOT / "training_data"), help="输出目录")
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 条（调试用）")
    parser.add_argument("--max_queries_per_doc", type=int, default=8, help="每条矿物最多生成多少查询")
    parser.add_argument("--num_negatives", type=int, default=7, help="每条查询的负样本数")
    parser.add_argument("--reranker_negatives", type=int, default=6, help="Reranker 使用的负样本数")
    parser.add_argument("--hard_ratio", type=float, default=0.7, help="难负样本占比")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding_model_path", default=str(PROJECT_ROOT / "model/Qwen3-Embedding-0.6B"))
    parser.add_argument("--no_embedding_mine", action="store_true", help="不加载 Embedding 模型，随机采负样本")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60, flush=True)
    print("1. 加载并规范化数据", flush=True)
    records = load_records(args.input)
    if args.limit:
        records = records[:args.limit]
    docs = [build_document(record, idx) for idx, record in enumerate(records)]
    docs = [doc for doc in docs if doc["name"] and doc["text"]]
    print(f"   有效文档: {len(docs)}", flush=True)

    attr_coverage = Counter()
    for doc in docs:
        for key in doc["attrs"]:
            attr_coverage[key] += 1
    print(f"   属性覆盖: {dict(attr_coverage)}", flush=True)

    print("2. 生成查询", flush=True)
    english_counts = build_english_counts(docs)
    attr_only = build_attribute_only_queries(docs)
    samples: List[Dict] = []
    query_types = Counter()
    for doc in docs:
        rng = random.Random(args.seed + doc["id"])
        english_unique = bool(doc["english_name"]) and english_counts[doc["english_name"].lower()] == 1
        queries = generate_queries(
            doc=doc,
            rng=rng,
            english_unique=english_unique,
            attr_only=attr_only.get(doc["id"], []),
            max_queries=args.max_queries_per_doc,
        )
        for query in queries:
            samples.append(
                {
                    "query": query["query"],
                    "query_type": query["query_type"],
                    "terms": query["terms"],
                    "positive_id": doc["id"],
                }
            )
            query_types[query["query_type"]] += 1
    print(f"   查询总数: {len(samples)}", flush=True)
    print(f"   查询类型: {dict(query_types)}", flush=True)

    print("3. 挖掘难负样本", flush=True)
    query_vectors = doc_vectors = None
    if not args.no_embedding_mine:
        print(f"   加载 Embedding 模型: {args.embedding_model_path}", flush=True)
        miner = Qwen3EmbeddingMiner(args.embedding_model_path, device=args.device)
        doc_vectors = miner.encode([doc["text"] for doc in docs], max_length=256)
        query_vectors = miner.encode([sample["query"] for sample in samples], is_query=True, max_length=96)
        print(f"   文档向量: {doc_vectors.shape}, 查询向量: {query_vectors.shape}", flush=True)
    negatives = mine_negatives(
        docs=docs,
        samples=samples,
        query_vectors=query_vectors,
        doc_vectors=doc_vectors,
        num_negatives=args.num_negatives,
        hard_ratio=args.hard_ratio,
        seed=args.seed,
    )

    print("4. 分组切分并写出", flush=True)
    split_of = assign_splits(docs, args.val_ratio, args.test_ratio, args.seed)

    corpus_rows = []
    for doc in docs:
        corpus_rows.append(
            {
                "id": doc["id"],
                "name": doc["name"],
                "english_name": doc["english_name"],
                "components": doc["components"],
                "text": doc["text"],
                "brief": doc["brief"],
                "main_purpose": doc["main_purpose"],
                "specific_purpose": doc["specific_purpose"],
                "origin": doc["origin"],
                "attrs": doc["attrs"],
                "split": split_of[doc["id"]],
            }
        )
    write_jsonl(output_dir / "corpus.jsonl", corpus_rows)

    embedding_splits = {"train": [], "val": [], "test": []}
    reranker_splits = {"train": [], "val": [], "test": []}
    pair_rows = {"train": [], "val": [], "test": []}
    split_query_counts = {"train": 0, "val": 0, "test": 0}

    for sample, neg_ids in zip(samples, negatives):
        positive = docs[sample["positive_id"]]
        split = split_of[positive["id"]]
        split_query_counts[split] += 1
        neg_docs = [docs[idx] for idx in neg_ids[:args.num_negatives]]
        negative_texts = [doc["text"] for doc in neg_docs]

        embedding_splits[split].append(
            {
                "query": sample["query"],
                "positive": positive["text"],
                "negatives": negative_texts,
                "query_type": sample["query_type"],
                "positive_id": positive["id"],
                "negative_ids": [doc["id"] for doc in neg_docs],
            }
        )

        reranker_negs = negative_texts[:args.reranker_negatives]
        reranker_splits[split].append(
            {
                "query": sample["query"],
                "response": positive["text"],
                "rejected_response": reranker_negs,
                "query_type": sample["query_type"],
                "positive_id": positive["id"],
            }
        )

        pair_rows[split].append(
            {
                "query": sample["query"],
                "doc": positive["text"],
                "label": 1,
                "query_type": sample["query_type"],
                "doc_id": positive["id"],
            }
        )
        for neg_doc in neg_docs[:args.reranker_negatives]:
            pair_rows[split].append(
                {
                    "query": sample["query"],
                    "doc": neg_doc["text"],
                    "label": 0,
                    "query_type": sample["query_type"],
                    "doc_id": neg_doc["id"],
                }
            )

    for split in ("train", "val", "test"):
        write_json(output_dir / "embedding" / f"{split}.json", embedding_splits[split])
        write_json(output_dir / "reranker" / f"{split}.json", reranker_splits[split])
        write_jsonl(output_dir / "reranker" / f"{split}_pairs.jsonl", pair_rows[split])

    print("5. 校验", flush=True)
    errors = validate(docs, samples, negatives, split_of)
    if errors:
        print(f"   ❌ 发现 {len(errors)} 个问题:", flush=True)
        for error in errors[:20]:
            print(f"      - {error}", flush=True)
    else:
        print("   ✅ 校验通过：无正负样本冲突、无重复查询、分组切分无泄漏", flush=True)

    stats = {
        "source": args.input,
        "num_docs": len(docs),
        "num_groups": len({doc["group"] for doc in docs}),
        "num_queries": len(samples),
        "query_types": dict(query_types),
        "attribute_coverage": dict(attr_coverage),
        "splits": {
            split: {
                "docs": sum(1 for doc in docs if split_of[doc["id"]] == split),
                "queries": split_query_counts[split],
                "embedding_samples": len(embedding_splits[split]),
                "reranker_samples": len(reranker_splits[split]),
                "reranker_pairs": len(pair_rows[split]),
            }
            for split in ("train", "val", "test")
        },
        "avg_negatives": round(float(np.mean([len(x) for x in negatives])), 2) if negatives else 0,
        "config": {
            "max_queries_per_doc": args.max_queries_per_doc,
            "num_negatives": args.num_negatives,
            "reranker_negatives": args.reranker_negatives,
            "hard_ratio": args.hard_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "seed": args.seed,
            "embedding_mining": not args.no_embedding_mine,
        },
        "validation_errors": len(errors),
    }
    write_json(output_dir / "stats.json", stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
    print(f"✅ 数据集已生成: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
