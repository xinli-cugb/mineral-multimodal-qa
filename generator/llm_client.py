"""
LLM 客户端封装
支持 Qwen2.5 和 ChatGLM3 两种大语言模型的调用
"""
import os
import sys
import json
import copy
from typing import List, Dict, Optional, Union, Generator, Any
from dataclasses import dataclass
import logging
import requests

from pathlib import Path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== 数据结构 ====================

@dataclass
class LLMResponse:
    """
    LLM 响应数据结构
    """
    content: str
    finish_reason: str = "stop"
    usage: Dict[str, int] = None
    model: str = ""

    def __str__(self) -> str:
        return self.content


@dataclass
class ChatMessage:
    """
    聊天消息数据结构
    """
    role: str
    content: str

    def to_dict(self) -> Dict:
        return {"role": self.role, "content": self.content}


# ==================== 直接加载的 LLM 客户端 ====================

class DirectLLMClient:
    """
    直接使用 transformers 加载模型的 LLM 客户端
    """

    def __init__(
        self,
        model_path: str,
        device: str = None,
        temperature: float = 0.65,
        max_tokens: int = 1024,
        top_p: float = 0.7,
        repetition_penalty: float = 1.1,
        stream: bool = False
    ):
        """
        初始化直接加载的 LLM 客户端

        Args:
            model_path: 模型路径或 HuggingFace Hub 名称
            device: 运行设备
            temperature: 温度参数
            max_tokens: 最大生成长度
            top_p: 核采样参数
            repetition_penalty: 重复惩罚
            stream: 是否流式输出
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_path = model_path
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens
        self.default_top_p = top_p
        self.default_repetition_penalty = repetition_penalty
        self.default_stream = stream

        logger.info(f"📦 加载 LLM 模型: {model_path}")
        logger.info(f"   设备: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )

        self.model.eval()

        logger.info(f"✅ LLM 模型加载成功")

    def chat(
        self,
        messages: List[Dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        stream: Optional[bool] = None,
        **kwargs
    ) -> LLMResponse:
        """
        聊天接口
        """
        temperature = temperature or self.default_temperature
        max_tokens = max_tokens or self.default_max_tokens
        top_p = top_p or self.default_top_p
        stream = stream if stream is not None else self.default_stream

        if repetition_penalty is None:
            repetition_penalty = self.default_repetition_penalty

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        return LLMResponse(
            content=response,
            finish_reason="stop",
            usage={
                "prompt_tokens": inputs.input_ids.shape[1],
                "completion_tokens": len(outputs[0]) - inputs.input_ids.shape[1],
                "total_tokens": len(outputs[0])
            },
            model=self.model_path
        )


# ==================== API 调用的 LLM 客户端 ====================

class LLMClient:
    """
    LLM 客户端基类 (API 模式)
    """

    def __init__(
        self,
        model_name: str,
        api_base: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        temperature: float = 0.65,
        max_tokens: int = 1024,
        top_p: float = 0.7,
        repetition_penalty: float = 1.1,
        stream: bool = False
    ):
        self.model_name = model_name
        self.api_base = api_base
        self.api_key = api_key
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens
        self.default_top_p = top_p
        self.default_repetition_penalty = repetition_penalty
        self.default_stream = stream

        logger.info(f"✅ LLM 客户端初始化: {model_name}")
        logger.info(f"   API 地址: {api_base}")

    def chat(
        self,
        messages: List[Dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        stream: Optional[bool] = None,
        **kwargs
    ) -> Union[LLMResponse, Generator[str, None, None]]:
        temperature = temperature or self.default_temperature
        max_tokens = max_tokens or self.default_max_tokens
        top_p = top_p or self.default_top_p
        stream = stream if stream is not None else self.default_stream

        try:
            from openai import OpenAI
            self.openai_client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base
            )

            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stream=stream,
                **kwargs
            )

            if stream:
                return self._stream_generator(response)
            else:
                choice = response.choices[0]
                return LLMResponse(
                    content=choice.message.content,
                    finish_reason=choice.finish_reason,
                    usage={
                        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                        "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                        "total_tokens": response.usage.total_tokens if response.usage else 0
                    },
                    model=self.model_name
                )

        except Exception as e:
            logger.error(f"API 调用失败: {e}")
            raise

    def _stream_generator(self, response):
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


# ==================== 工厂函数 ====================

def create_llm_client(
    model_type: str = "qwen2.5",
    model_path: str = None,
    use_direct: bool = True,
    **kwargs
) -> Union[LLMClient, DirectLLMClient]:
    """
    创建 LLM 客户端

    Args:
        model_type: 模型类型 ("qwen2.5" 或 "chatglm3")
        model_path: 模型路径
        use_direct: 是否直接加载模型（不依赖 API 服务）
        **kwargs: 其他参数
    """
    if model_path is None:
        if model_type == "qwen2.5":
            model_path = "Qwen/Qwen2.5-7B-Instruct"
        elif model_type == "chatglm3":
            model_path = "THUDM/chatglm3-6b"
        else:
            model_path = "Qwen/Qwen2.5-7B-Instruct"

    if use_direct:
        return DirectLLMClient(
            model_path=model_path,
            **kwargs
        )
    else:
        return LLMClient(
            model_name=model_path,
            **kwargs
        )


def load_qwen_client(
    model_path: str = "Qwen/Qwen2.5-7B-Instruct",
    use_direct: bool = True,
    **kwargs
) -> Union[LLMClient, DirectLLMClient]:
    """
    加载 Qwen2.5 客户端
    """
    return create_llm_client(
        model_type="qwen2.5",
        model_path=model_path,
        use_direct=use_direct,
        **kwargs
    )


if __name__ == "__main__":
    print("=" * 60)
    print("LLM 客户端测试 (直接加载模式)")
    print("=" * 60)

    try:
        client = load_qwen_client(
            model_path="/root/V2_0_Mineral_QA/model/Qwen2.5-7B-Instruct",
            use_direct=True
        )

        messages = [
            {"role": "system", "content": "你是一个专业的矿物学助手。"},
            {"role": "user", "content": "石英的化学成分是什么？"}
        ]

        print("\n📤 发送消息:")
        print(f"   {messages}")

        response = client.chat(messages, stream=False)

        print(f"\n📥 收到回复:")
        print(f"   {response.content}")
        print(f"   Usage: {response.usage}")

    except Exception as e:
        print(f"\n⚠️ 测试失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n✅ 测试完成!")
