
import json
import os
import time
from typing import Dict, List
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI


class MineralDataRefactor:

    
    def __init__(
        self,
        api_key: str = "ms-dfd7ef08-e098-42f7-a210-cd0d78fc40f0",
        base_url: str = "https://api-inference.modelscope.cn/v1",
        model: str = "ZhipuAI/GLM-5.1"
    ):
        """
        初始化重构器
        
        Args:
            api_key: ModelScope API Key
            base_url: API 基础 URL
            model: 模型名称
        """
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self.model = model
    
    def refactor_mineral_data(self, mineral_data: Dict, max_retries: int = 3) -> Dict:
        """
        重构单条矿物数据
        
        Args:
            mineral_data: 原始矿物数据
            max_retries: 最大重试次数
        
        Returns:
            重构后的矿物数据
        """
        # 提取原始字段
        name = mineral_data.get('name', '')
        english_name = mineral_data.get('english_name', '')
        main_purpose = mineral_data.get('main_purpose', '')
        brief_characterization = mineral_data.get('brief_characterization', '')
        specific_purpose = mineral_data.get('specific_purpose', '')
        
        # 构建提示词
        prompt = f"""请根据以下矿物信息，将主要用途、简要特征、具体用途合并成一段连贯的自然语言描述：

矿物名称：{name}
英文名称：{english_name}
主要用途：{main_purpose}
简要特征：{brief_characterization}
具体用途：{specific_purpose}

要求：
- 将主要用途、简要特征、具体用途合并成一段连贯的自然语言描述
- 使用专业但易懂的语言
- 保持信息的完整性和准确性
- 形成流畅的段落，避免简单罗列
- 描述要详细，充分展开每个方面的信息

请直接输出描述内容，不要包含任何标题或格式标记。"""

        # 重试机制
        for attempt in range(max_retries):
            try:
                # 调用 API - 必须使用流式响应
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            'role': 'system',
                            'content': '你是一个专业的矿物学专家，擅长将矿物数据改写为自然语言描述。'
                        },
                        {
                            'role': 'user',
                            'content': prompt
                        }
                    ],
                    stream=True,  # 必须使用流式响应
                    temperature=0.7,
                    max_tokens=2000
                )
                
                # 处理流式响应
                refined_text = ""
                done_reasoning = False
                
                for chunk in response:
                    if chunk.choices:
                        reasoning_chunk = chunk.choices[0].delta.reasoning_content
                        answer_chunk = chunk.choices[0].delta.content
                        
                        # 处理推理内容（如果有）
                        if reasoning_chunk and reasoning_chunk != '':
                            # 推理内容，可以选择忽略或记录
                            pass
                        
                        # 处理实际内容
                        if answer_chunk and answer_chunk != '':
                            if not done_reasoning:
                                done_reasoning = True
                            refined_text += answer_chunk
                
                # 检查是否获得了内容
                if not refined_text:
                    raise ValueError("API 返回了空的内容")
                
                # 构建重构后的数据
                refactored_data = {
                    'name': name,
                    'English_name': english_name,
                    '描述': refined_text
                }
                
                return refactored_data
                
            except Exception as e:
                error_msg = str(e)
                print(f"处理 {name} 时出错 (尝试 {attempt + 1}/{max_retries}): {error_msg}")
                
                # 如果是速率限制错误，等待更长时间
                if '429' in error_msg or 'rate limit' in error_msg.lower():
                    wait_time = min(60, 10 * (2 ** attempt))  # 指数退避，最多60秒
                    print(f"遇到速率限制，等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    # 其他错误，如果是最后一次尝试，返回简单版本
                    if attempt == max_retries - 1:
                        simple_description = f"{main_purpose}。{brief_characterization}。{specific_purpose}"
                        return {
                            'name': name,
                            'English_name': english_name,
                            '描述': simple_description,
                            'error': error_msg
                        }
                    else:
                        # 短暂等待后重试
                        time.sleep(5)
        
        # 如果所有重试都失败，返回简单版本
        simple_description = f"{main_purpose}。{brief_characterization}。{specific_purpose}"
        return {
            'name': name,
            'English_name': english_name,
            '描述': simple_description,
            'error': 'All retries failed'
        }
    
    def refactor_batch(
        self,
        data_list: List[Dict],
        output_file: str,
        batch_size: int = 10,
        delay: float = 1.0
    ):
        """
        批量重构数据
        
        Args:
            data_list: 数据列表
            output_file: 输出文件路径
            batch_size: 批次大小
            delay: 请求间隔（秒）
        """
        refactored_data = []
        
        # 创建输出目录
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 处理数据
        for i, mineral_data in enumerate(tqdm(data_list, desc="重构矿物数据")):
            # 重构单条数据
            refactored = self.refactor_mineral_data(mineral_data)
            refactored_data.append(refactored)
            
            # 保存中间结果
            if (i + 1) % batch_size == 0:
                self._save_data(refactored_data, output_file)
                print(f"\n已处理 {i + 1}/{len(data_list)} 条数据，已保存到 {output_file}")
            
            # 延迟以避免 API 限流
            time.sleep(delay)
        
        # 保存最终结果
        self._save_data(refactored_data, output_file)
        print(f"\n✅ 数据重构完成！共处理 {len(refactored_data)} 条数据")
        print(f"结果已保存到: {output_file}")
    
    def _save_data(self, data: List[Dict], output_file: str):
        """保存数据到文件"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="矿物数据重构脚本")
    parser.add_argument("--input_file", type=str, 
                        default="/root/V2_0_Mineral_QA/text_data/all_v1.json",
                        help="输入文件路径")
    parser.add_argument("--output_file", type=str, 
                        default="/root/V2_0_Mineral_QA/text_data/all_v1_refactored.json",
                        help="输出文件路径")
    parser.add_argument("--api_key", type=str, required=True,
                        help="ModelScope API Key")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="批次大小")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="请求间隔（秒）")
    parser.add_argument("--limit", type=int, default=None,
                        help="处理数据数量限制（用于测试）")
    
    args = parser.parse_args()
    
    # 使用原始 API Key，不进行任何处理
    api_key = args.api_key.strip()
    print(f"使用 API Key: {api_key[:15]}...{api_key[-15:]}")
    
    # 读取原始数据
    print(f"读取数据: {args.input_file}")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        # 处理多行 JSON 格式（每行一个 JSON 对象）
        data = []
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"跳过无效行: {line[:50]}... 错误: {e}")
    
    print(f"共 {len(data)} 条矿物数据")
    
    # 限制处理数量（用于测试）
    if args.limit:
        data = data[:args.limit]
        print(f"限制处理数量为: {args.limit}")
    
    # 创建重构器
    refactor = MineralDataRefactor(
        api_key=api_key,
        model="ZhipuAI/GLM-5.1"
    )
    
    # 批量重构
    refactor.refactor_batch(
        data_list=data,
        output_file=args.output_file,
        batch_size=args.batch_size,
        delay=args.delay
    )


if __name__ == "__main__":
    main()