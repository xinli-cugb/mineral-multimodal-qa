"""
使用本地 Qwen2.5-7B-Instruct 模型重构矿物数据
"""
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
import argparse


class LocalMineralDataRefactor:
    """
    本地矿物数据重构器
    
    使用本地 Qwen2.5-7B-Instruct 模型将矿物数据重构为指定格式：
    {
        "name": "矿物名称",
        "English_name": "英文名称", 
        "描述": "将主要用途、简要特征、具体用途合并成的自然语言描述"
    }
    """
    
    def __init__(
        self,
        model_path: str = "/root/V2_0_Mineral_QA/model/Qwen2.5-7B-Instruct",
        device: str = "cuda"
    ):
        """
        初始化本地重构器
        
        Args:
            model_path: 本地模型路径
            device: 运行设备 (cuda/cpu)
        """
        self.model_path = model_path
        self.device = device
        
        print(f"加载模型: {model_path}")
        print(f"使用设备: {device}")
        
        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )
        
        if device == "cpu":
            self.model = self.model.to(device)
        
        print("✅ 模型加载完成！")
    
    def refactor_mineral_data(self, mineral_data: Dict) -> Dict:
        """
        重构单条矿物数据
        
        Args:
            mineral_data: 原始矿物数据
        
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
        
        try:
            # 使用 Qwen 的聊天模板
            messages = [
                {"role": "system", "content": "你是一个专业的矿物学专家，擅长将矿物数据改写为自然语言描述。"},
                {"role": "user", "content": prompt}
            ]
            
            # 应用聊天模板
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # 编码输入
            model_inputs = self.tokenizer(
                [text],
                return_tensors="pt"
            ).to(self.device)
            
            # 生成响应
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=1000,
                temperature=0.7,
                top_p=0.8,
                do_sample=True,
                repetition_penalty=1.1
            )
            
            # 解码响应
            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            refined_text = self.tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0].strip()
            
            # 构建重构后的数据
            refactored_data = {
                'name': name,
                'English_name': english_name,
                '描述': refined_text
            }
            
            return refactored_data
            
        except Exception as e:
            print(f"处理 {name} 时出错: {e}")
            # 返回原始数据的简单版本
            simple_description = f"{main_purpose}。{brief_characterization}。{specific_purpose}"
            return {
                'name': name,
                'English_name': english_name,
                '描述': simple_description,
                'error': str(e)
            }
    
    def refactor_batch(
        self,
        data_list: List[Dict],
        output_file: str,
        batch_size: int = 10
    ):
        """
        批量重构数据
        
        Args:
            data_list: 数据列表
            output_file: 输出文件路径
            batch_size: 批次大小
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
        
        # 保存最终结果
        self._save_data(refactored_data, output_file)
        print(f"\n✅ 数据重构完成！共处理 {len(refactored_data)} 条数据")
        print(f"结果已保存到: {output_file}")
        
        # 统计错误
        errors = [d for d in refactored_data if 'error' in d]
        if errors:
            print(f"⚠️  有 {len(errors)} 条数据处理出错")
    
    def _save_data(self, data: List[Dict], output_file: str):
        """保存数据到文件"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="本地矿物数据重构脚本")
    parser.add_argument("--input_file", type=str, 
                        default="/root/V2_0_Mineral_QA/text_data/all_v1.json",
                        help="输入文件路径")
    parser.add_argument("--output_file", type=str, 
                        default="/root/V2_0_Mineral_QA/text_data/all_v1_refactored.json",
                        help="输出文件路径")
    parser.add_argument("--model_path", type=str, 
                        default="/root/V2_0_Mineral_QA/model/Qwen2.5-7B-Instruct",
                        help="本地模型路径")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="运行设备")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="批次大小")
    parser.add_argument("--limit", type=int, default=None,
                        help="处理数据数量限制（用于测试）")
    
    args = parser.parse_args()
    
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
    
    # 创建本地重构器
    refactor = LocalMineralDataRefactor(
        model_path=args.model_path,
        device=args.device
    )
    
    # 批量重构
    refactor.refactor_batch(
        data_list=data,
        output_file=args.output_file,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()