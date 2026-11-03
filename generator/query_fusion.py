"""

输入：用户问题 + Swin Top-K 视觉锚点（矿物名、置信度、排名）
输出：面向检索的融合查询（意图、标准矿物名、消歧约束）

"""
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

QUERY_FUSION_SYSTEM_PROMPT = """你是矿物岩石领域的检索查询改写器。你的任务是把"用户问题 + 图像识别出的候选矿物"改写成一条用于知识库检索的查询，而不是回答问题。

必须遵守以下规则：
1. 意图判定：把问题归入 definition（定义/是什么）、property（物性/成分/硬度/颜色等）、usage（用途）、origin（产地/成因）、identification（鉴别）之一。
2. 锚点使用：rank 1 的矿物是主锚点；其余锚点只能作为候选或消歧线索。置信度低的锚点不得当作确定事实。
3. 命名规范：只使用标准中文矿物名，可附带英文名；不得臆造、音译或合并矿物名称。
4. 信息边界：不得把锚点的性质（成分、硬度、颜色等）写进查询，除非用户问题本身已经包含；信息不足时保持中性表述，并令 needs_clarification 为 true。
5. 改写后的查询要简短（不超过 60 个汉字），保留用户原始意图与关键限定词，适合向量检索与 BM25 检索。

只输出一个 JSON 对象，不要输出解释或 Markdown：
{"intent": "...", "minerals": [{"name": "标准中文名", "english_name": "英文名", "confidence": 0.0}], "fused_query": "改写后的检索查询", "needs_clarification": false}"""


@dataclass
class FusionResult:
    fused_query: str
    intent: str = "unknown"
    minerals: List[Dict] = field(default_factory=list)
    needs_clarification: bool = False
    used_llm: bool = False

    def to_dict(self) -> Dict:
        return {
            "fused_query": self.fused_query,
            "intent": self.intent,
            "minerals": self.minerals,
            "needs_clarification": self.needs_clarification,
            "used_llm": self.used_llm,
        }


def _extract_json(text: str) -> Optional[Dict]:
    """从模型输出中提取第一个 JSON 对象"""
    text = re.sub(r"```(?:json)?", "", text or "").strip()
    text = text.strip("`").strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


class QueryFuser:
    """基于本地 LLM 的查询融合器，失败时退化为确定性拼接"""

    def __init__(
        self,
        llm_client,
        max_anchors: int = 3,
        max_tokens: int = 200,
        temperature: float = 0.1,
    ):
        self.llm_client = llm_client
        self.max_anchors = max_anchors
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _fallback(self, query: str, anchors: Optional[List[Dict]]) -> FusionResult:
        names: List[str] = []
        for anchor in (anchors or [])[:self.max_anchors]:
            name = str(anchor.get("mineral_name", "")).strip()
            if name and name not in names:
                names.append(name)
        fused = (" ".join(names) + " " + query).strip() if names else query
        return FusionResult(fused_query=fused, used_llm=False)

    def _build_user_prompt(self, query: str, anchors: Optional[List[Dict]]) -> str:
        lines = [f"用户问题：{query}", "视觉识别候选："]
        if not anchors:
            lines.append("（无图像输入，仅改写文本问题）")
        else:
            for anchor in anchors[:self.max_anchors]:
                lines.append(
                    "- {name}（英文：{en}），置信度 {conf:.3f}，rank {rank}".format(
                        name=anchor.get("mineral_name", ""),
                        en=anchor.get("english_name", "") or "未知",
                        conf=float(anchor.get("confidence", 0.0)),
                        rank=anchor.get("rank", 0),
                    )
                )
        return "\n".join(lines)

    def fuse(self, query: str, anchors: Optional[List[Dict]] = None) -> FusionResult:
        """生成融合查询；任何异常都退回确定性拼接，保证 pipeline 不中断"""
        if self.llm_client is None:
            return self._fallback(query, anchors)

        messages = [
            {"role": "system", "content": QUERY_FUSION_SYSTEM_PROMPT},
            {"role": "user", "content": self._build_user_prompt(query, anchors)},
        ]

        try:
            response = self.llm_client.chat(
                messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=False,
            )
            content = response.content if hasattr(response, "content") else str(response)
            obj = _extract_json(content)
            if not obj:
                logger.warning("查询融合输出无法解析为 JSON，回退规则拼接")
                return self._fallback(query, anchors)

            fused = str(obj.get("fused_query", "")).strip()
            if not fused:
                logger.warning("查询融合结果为空，回退规则拼接")
                return self._fallback(query, anchors)

            return FusionResult(
                fused_query=fused,
                intent=str(obj.get("intent", "unknown")),
                minerals=list(obj.get("minerals") or []),
                needs_clarification=bool(obj.get("needs_clarification", False)),
                used_llm=True,
            )
        except Exception as exc:
            logger.warning(f"查询融合失败，回退规则拼接: {exc}")
            return self._fallback(query, anchors)
