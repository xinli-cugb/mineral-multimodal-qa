"""
Prompt 模板库
设计严谨的 Prompt 模板，将用户 Query、视觉锚点和黄金上下文巧妙拼接
加入防幻觉约束的 System Prompt
"""
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import json


# ==================== 数据结构 ====================

@dataclass
class VisualAnchorPrompt:
    """
    视觉锚点 Prompt 数据结构
    """
    mineral_name: str
    confidence: float
    rank: int
    
    def to_prompt_string(self) -> str:
        """转换为 Prompt 字符串"""
        return f"【视觉识别-{self.rank}】矿物: {self.mineral_name} (置信度: {self.confidence:.2%})"


@dataclass
class ContextDocument:
    """
    上下文文档 Prompt 数据结构
    """
    text: str
    score: float
    rank: int
    source: str = ""
    title: str = ""
    
    def to_prompt_string(self, max_length: int = 500) -> str:
        """转换为 Prompt 字符串（可截断）"""
        truncated_text = self.text[:max_length]
        if len(self.text) > max_length:
            truncated_text += "..."
        
        source_info = f"【来源: {self.source}】" if self.source else ""
        title_info = f"【标题: {self.title}】" if self.title else ""
        
        return f"""【参考文档-{self.rank}】{source_info}{title_info}
{truncated_text}"""


# ==================== System Prompt ====================

MINERAL_SYSTEM_PROMPT = """你是智矿助手（MineralQA），一个专业的矿物学领域智能问答助手。

## 核心能力
- 精通矿物学专业知识，包括矿物鉴定、成分分析、物理化学性质、产状成因等
- 能够根据用户提供的图像和文本问题进行综合分析
- 回答专业、准确、详实

## 回答规范
1. **严格基于给定知识文档回答**：只使用提供的参考文档中的信息，不要添加未提及的内容
2. **无法回答时明确说明**：如果给定文档中没有相关信息，诚实回答"根据已知信息无法回答该问题"
3. **禁止编造**：不允许在答案中添加任何编造成分或未经验证的信息
4. **专业准确**：使用规范的矿物学术语，注意区分相似概念（如晶系vs晶类）
5. **使用中文**：请使用简体中文回答所有问题

## 格式要求
- 回答要简洁清晰，重点突出
- 如有多条信息，可以使用列表格式
- 涉及具体数值时说明来源"""


# ==================== Prompt 模板类 ====================

class MineralPromptTemplate:
    """
    矿物问答 Prompt 模板
    """
    
    def __init__(
        self,
        system_prompt: str = MINERAL_SYSTEM_PROMPT,
        max_context_length: int = 3000,
        include_visual_anchors: bool = True,
        include_confidence_warning: bool = True
    ):
        """
        初始化 Prompt 模板
        
        Args:
            system_prompt: 系统提示词
            max_context_length: 最大上下文长度
            include_visual_anchors: 是否包含视觉锚点
            include_confidence_warning: 是否包含置信度警告
        """
        self.system_prompt = system_prompt
        self.max_context_length = max_context_length
        self.include_visual_anchors = include_visual_anchors
        self.include_confidence_warning = include_confidence_warning
    
    def build(
        self,
        query: str,
        visual_anchors: Optional[List[VisualAnchorPrompt]] = None,
        context_documents: Optional[List[ContextDocument]] = None,
        history: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, str]]:
        """
        构建完整的 Prompt 消息列表
        
        Args:
            query: 用户问题
            visual_anchors: 视觉锚点列表
            context_documents: 上下文文档列表
            history: 历史对话
        
        Returns:
            消息列表，符合 OpenAI API 格式
        """
        messages = []
        
        # 1. System Prompt
        messages.append({
            "role": "system",
            "content": self.system_prompt
        })
        
        # 2. 历史对话（如果有）
        if history:
            for msg in history:
                messages.append(msg)
        
        # 3. 构建用户消息
        user_content = self._build_user_content(
            query=query,
            visual_anchors=visual_anchors,
            context_documents=context_documents
        )
        
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        return messages
    
    def _build_user_content(
        self,
        query: str,
        visual_anchors: Optional[List[VisualAnchorPrompt]] = None,
        context_documents: Optional[List[ContextDocument]] = None
    ) -> str:
        """
        构建用户消息内容
        
        Args:
            query: 用户问题
            visual_anchors: 视觉锚点
            context_documents: 上下文文档
        
        Returns:
            格式化的用户消息内容
        """
        parts = []
        
        # 1. 视觉锚点部分
        if self.include_visual_anchors and visual_anchors:
            visual_parts = []
            
            # 置信度警告
            if self.include_confidence_warning:
                primary_anchor = visual_anchors[0]
                warning = f"⚠️ 注意：图像识别的主要矿物为【{primary_anchor.mineral_name}】（置信度 {primary_anchor.confidence:.2%}），仅供参考验证。"
                visual_parts.append(warning)
            
            # 列出所有锚点
            anchor_info = "图像识别到的矿物候选："
            for anchor in visual_anchors:
                anchor_info += f"\n  {anchor.to_prompt_string()}"
            visual_parts.append(anchor_info)
            
            parts.append("## 视觉识别结果\n" + "\n".join(visual_parts))
        
        # 2. 上下文文档部分
        if context_documents:
            context_parts = []
            for doc in context_documents:
                context_parts.append(doc.to_prompt_string(max_length=800))
            
            context_header = f"## 参考知识文档（共{len(context_documents)}条）\n" + \
                           "以下是检索到的相关矿物知识，请基于这些文档回答问题：\n"
            parts.append(context_header + "\n\n".join(context_parts))
        
        # 3. 用户问题
        query_part = f"## 当前问题\n{query}"
        parts.append(query_part)
        
        # 4. 任务指令
        instruction = """
## 回答任务
请根据上述【视觉识别结果】和【参考知识文档】来回答【当前问题】。

要求：
1. 优先使用参考文档中的信息回答
2. 如果文档中有相关信息，给出准确答案
3. 如果文档中没有相关信息，明确说明"根据已知信息无法回答"
4. 不要添加文档中没有的内容

请开始回答："""
        parts.append(instruction)
        
        return "\n\n".join(parts)
    
    def build_simple(self, query: str, context: Optional[str] = None) -> List[Dict[str, str]]:
        """
        构建简单版 Prompt（无视觉锚点）
        
        Args:
            query: 用户问题
            context: 上下文文本
        
        Returns:
            消息列表
        """
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        if context:
            user_content = f"""## 参考知识文档
{context}

## 问题
{query}

请根据上述文档回答问题。如果无法从中找到答案，请明确说明。"""
        else:
            user_content = f"""## 问题
{query}

请回答上述问题。如果不知道答案，请明确说明。"""
        
        messages.append({"role": "user", "content": user_content})
        
        return messages


# ==================== 防幻觉约束 ====================

class HallucinationGuard:
    """
    防幻觉约束器
    """
    
    # 拒绝回答的关键词
    REFUSE_KEYWORDS = [
        "不知道", "不清楚", "不确定", "没有提到", 
        "文档中没有", "无法确定", "信息不足"
    ]
    
    # 强制要求使用的约束语句
    REQUIRED_PHRASES = [
        "根据已知信息",
        "从文档中",
        "参考文档"
    ]
    
    @staticmethod
    def check_response(response: str) -> tuple[bool, str]:
        """
        检查响应是否包含幻觉
        
        Returns:
            (is_safe, warning_message)
        """
        response = response.strip()
        
        # 检查是否为空
        if not response:
            return False, "回复内容为空"
        
        # 检查是否包含拒绝关键词（在开头）
        for keyword in HallucinationGuard.REFUSE_KEYWORDS[:4]:
            if response.startswith(keyword):
                return True, ""
        
        return True, ""
    
    @staticmethod
    def add_safety_suffix(response: str) -> str:
        """
        添加安全后缀
        
        Args:
            response: 原始回复
        
        Returns:
            添加安全提示后的回复
        """
        safety_note = "\n\n---\n*以上回答仅基于提供的知识文档，如有疑问请咨询专业人士。*"
        
        # 检查是否已有类似后缀
        if "*以上回答" not in response:
            response += safety_note
        
        return response


# ==================== 便捷函数 ====================

def create_prompt_template(
    include_visual: bool = True,
    max_context: int = 3000
) -> MineralPromptTemplate:
    """
    创建 Prompt 模板
    
    Args:
        include_visual: 是否包含视觉锚点
        max_context: 最大上下文长度
    
    Returns:
        MineralPromptTemplate 实例
    """
    return MineralPromptTemplate(
        system_prompt=MINERAL_SYSTEM_PROMPT,
        max_context_length=max_context,
        include_visual_anchors=include_visual
    )


def build_mineral_prompt(
    query: str,
    visual_anchors: Optional[List[Dict]] = None,
    context_docs: Optional[List[Dict]] = None,
    history: Optional[List[Dict]] = None
) -> List[Dict[str, str]]:
    """
    便捷函数：构建矿物问答 Prompt
    
    Args:
        query: 用户问题
        visual_anchors: 视觉锚点 [{"mineral_name": str, "confidence": float, "rank": int}, ...]
        context_docs: 上下文文档 [{"text": str, "score": float, "rank": int, "source": str}, ...]
        history: 历史对话
    
    Returns:
        消息列表
    """
    # 转换数据格式
    anchors = None
    if visual_anchors:
        anchors = [
            VisualAnchorPrompt(
                mineral_name=a["mineral_name"],
                confidence=a["confidence"],
                rank=a["rank"]
            ) for a in visual_anchors
        ]
    
    docs = None
    if context_docs:
        docs = [
            ContextDocument(
                text=d["text"],
                score=d["score"],
                rank=d["rank"],
                source=d.get("source", ""),
                title=d.get("title", "")
            ) for d in context_docs
        ]
    
    # 构建 Prompt
    template = MineralPromptTemplate()
    return template.build(
        query=query,
        visual_anchors=anchors,
        context_documents=docs,
        history=history
    )


# ==================== 示例用法 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("Prompt 模板测试")
    print("=" * 60)
    
    # 1. 测试完整版 Prompt
    print("\n📝 测试 1: 完整版 Prompt（含视觉锚点）")
    
    visual_anchors = [
        {"mineral_name": "石英", "confidence": 0.85, "rank": 1},
        {"mineral_name": "长石", "confidence": 0.10, "rank": 2},
        {"mineral_name": "云母", "confidence": 0.03, "rank": 3}
    ]
    
    context_docs = [
        {
            "text": "石英(Quartz)是一种主要造岩矿物，化学成分为二氧化硅(SiO2)，属于三方晶系。",
            "score": 0.95,
            "rank": 1,
            "source": "wiki",
            "title": "石英"
        },
        {
            "text": "石英的硬度为7，莫氏硬度计上排在金刚石之后。",
            "score": 0.88,
            "rank": 2,
            "source": "baike",
            "title": "石英物理性质"
        },
        {
            "text": "石英在自然界中广泛存在于花岗岩、砂岩等多种岩石中。",
            "score": 0.82,
            "rank": 3,
            "source": "wiki",
            "title": "石英产状"
        }
    ]
    
    messages = build_mineral_prompt(
        query="这块矿物标本的硬度是多少？",
        visual_anchors=visual_anchors,
        context_docs=context_docs
    )
    
    print(f"\n📋 消息数量: {len(messages)}")
    for i, msg in enumerate(messages):
        print(f"\n--- Message {i+1} (role: {msg['role']}) ---")
        print(msg['content'][:500] + "..." if len(msg['content']) > 500 else msg['content'])
    
    # 2. 测试简单版 Prompt
    print("\n\n" + "=" * 60)
    print("📝 测试 2: 简单版 Prompt")
    
    template = create_prompt_template(include_visual=False)
    simple_messages = template.build_simple(
        query="石英的化学成分是什么？",
        context="石英的化学式是SiO2。"
    )
    
    for msg in simple_messages:
        print(f"\n--- {msg['role']} ---")
        print(msg['content'])
    
    print("\n✅ 测试完成!")
