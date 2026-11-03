"""
矿物图像问答系统启动脚本
支持上传矿物图片，通过 Swin Transformer 识别，查询数据库，然后使用大模型生成回答
基于 V2.0 矿物问答系统架构
"""
import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Optional
from PIL import Image

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
from vision.swin_model import MineralSwinClassifier
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class SimpleMineralQASystem:
    """
    简化版矿物图像问答系统
    
    完整流程：
    1. 上传矿物图片
    2. Swin Transformer 识别矿物
    3. 查询数据库获取详细信息
    4. 使用大模型生成自然语言回答
    """
    
    def __init__(
        self,
        swin_model_path: str = "/root/rag/models/checkpoints/swin_transformer/best_model_acc7886_ep10.pth",
        swin_id2label_path: str = "/root/V2_0_Mineral_QA/model/id2label.json",
        llm_model_path: str = "/root/V2_0_Mineral_QA/model/Qwen2.5-7B-Instruct",
        database_path: str = "/root/V2_0_Mineral_QA/text_data/45.json",
        device: str = "cuda"
    ):
        """
        初始化问答系统
        
        Args:
            swin_model_path: Swin 模型路径
            swin_id2label_path: Swin 标签映射文件
            llm_model_path: 大模型路径
            database_path: 矿物数据库路径
            device: 运行设备
        """
        self.device = device
        self.database_path = database_path
        
        print("=" * 60)
        print("矿物图像问答系统初始化")
        print("=" * 60)
        
        # 1. 加载 Swin Transformer 模型
        print("\n[1/3] 加载 Swin Transformer 模型...")
        try:
            self.swin_classifier = MineralSwinClassifier(
                model_path=swin_model_path,
                id2label_path=swin_id2label_path,
                device=device,
                top_k=3
            )
            print("✅ Swin 模型加载成功")
        except Exception as e:
            print(f"❌ Swin 模型加载失败: {e}")
            self.swin_classifier = None
        
        # 2. 加载矿物数据库
        print("\n[2/3] 加载矿物数据库...")
        self.mineral_database = self._load_database(database_path)
        print(f"✅ 数据库加载成功")
        
        # 3. 加载大语言模型
        print("\n[3/3] 加载大语言模型...")
        try:
            self.llm_tokenizer = AutoTokenizer.from_pretrained(
                llm_model_path,
                trust_remote_code=True
            )
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                llm_model_path,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True
            )
            
            if device == "cpu":
                self.llm_model = self.llm_model.to(device)
            
            self.llm_model.eval()
            print("✅ 大语言模型加载成功")
        except Exception as e:
            print(f"❌ 大语言模型加载失败: {e}")
            self.llm_model = None
            self.llm_tokenizer = None
        
        print("\n" + "=" * 60)
        print("系统初始化完成！")
        print("=" * 60)
    
    def _load_database(self, database_path: str) -> Dict[str, Dict]:
        """
        加载矿物数据库
        
        Args:
            database_path: 数据库文件路径
        
        Returns:
            矿物名称到矿物信息的映射
        """
        with open(database_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 构建名称到信息的映射
        mineral_dict = {}
        for item in data:
            name = item.get('name', '')
            if name:
                mineral_dict[name] = item
        
        return mineral_dict
    
    def identify_mineral(self, image: Image.Image) -> Dict:
        """
        识别矿物图像
        
        Args:
            image: PIL 图像对象
        
        Returns:
            识别结果
        """
        if self.swin_classifier is None:
            return {
                "success": False,
                "error": "Swin 模型未加载"
            }
        
        try:
            # 获取视觉锚点
            result = self.swin_classifier.get_visual_anchors(image)
            
            # 提取 Top-3 预测
            predictions = []
            for anchor in result.get("visual_anchors", []):
                predictions.append({
                    "rank": anchor["rank"],
                    "mineral_name": anchor["mineral_name"],
                    "confidence": anchor["confidence"]
                })
            
            return {
                "success": True,
                "predictions": predictions,
                "total_predictions": result.get("total_predictions", 0)
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def query_database(self, mineral_name: str) -> Optional[Dict]:
        """
        查询矿物数据库
        
        Args:
            mineral_name: 矿物名称（支持中文或英文）
        
        Returns:
            矿物信息
        """
        # 1. 精确匹配中文名称
        if mineral_name in self.mineral_database:
            return self.mineral_database[mineral_name]
        
        # 2. 精确匹配英文名称
        for name, info in self.mineral_database.items():
            english_name = info.get('english_name', '')
            if english_name and english_name.lower() == mineral_name.lower():
                return info
        
        # 3. 模糊匹配中文名称（去除空格和括号）
        normalized_name = mineral_name.strip().replace(" ", "")
        for name, info in self.mineral_database.items():
            if name.replace(" ", "") == normalized_name:
                return info
        
        # 4. 模糊匹配英文名称
        for name, info in self.mineral_database.items():
            english_name = info.get('english_name', '')
            if english_name and normalized_name in english_name.lower():
                return info
        
        # 5. 部分匹配中文名称
        for name, info in self.mineral_database.items():
            if mineral_name in name or name in mineral_name:
                return info
        
        return None
    
    def generate_answer(
        self,
        mineral_name: str,
        mineral_info: Dict,
        user_question: str = "请介绍一下这种矿物",
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> str:
        """
        使用大模型生成回答
        
        Args:
            mineral_name: 矿物名称
            mineral_info: 矿物信息
            user_question: 用户问题
            temperature: 生成温度
            max_tokens: 最大生成长度
        
        Returns:
            生成的回答
        """
        if self.llm_model is None or self.llm_tokenizer is None:
            return "大语言模型未加载，无法生成详细回答。"
        
        # 构建提示词
        name = mineral_info.get('name', mineral_name)
        english_name = mineral_info.get('english_name', '')
        text = mineral_info.get('text', '')
        
        prompt = f"""你是一个专业的矿物学专家。根据以下矿物信息，回答用户的问题。

矿物名称：{name}
英文名称：{english_name}

详细信息：
{text}

用户问题：{user_question}

请用专业但易懂的语言回答，重点突出用户关心的方面。如果信息不足，请诚实地说明。"""
        
        try:
            # 使用 Qwen 的聊天模板
            messages = [
                {"role": "system", "content": "你是一个专业的矿物学专家，擅长矿物识别和解释。"},
                {"role": "user", "content": prompt}
            ]
            
            # 应用聊天模板
            text = self.llm_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # 编码输入
            model_inputs = self.llm_tokenizer(
                [text],
                return_tensors="pt"
            ).to(self.device)
            
            # 生成响应
            with torch.no_grad():
                generated_ids = self.llm_model.generate(
                    model_inputs.input_ids,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.8,
                    do_sample=True,
                    repetition_penalty=1.1
                )
            
            # 解码响应
            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            answer = self.llm_tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0].strip()
            
            return answer
        
        except Exception as e:
            return f"生成回答时出错: {str(e)}"
    
    def process_image(
        self,
        image: Image.Image,
        user_question: str = "请介绍一下这种矿物",
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> Dict:
        """
        完整处理流程：识别 -> 查询 -> 生成回答
        
        Args:
            image: 矿物图像
            user_question: 用户问题
            temperature: 生成温度
            max_tokens: 最大生成长度
        
        Returns:
            完整的处理结果
        """
        start_time = time.time()
        
        # 1. 识别矿物
        identification_result = self.identify_mineral(image)
        
        if not identification_result["success"]:
            return {
                "success": False,
                "error": identification_result.get("error", "识别失败"),
                "total_time": time.time() - start_time
            }
        
        predictions = identification_result["predictions"]
        
        if not predictions:
            return {
                "success": False,
                "error": "未能识别出矿物",
                "predictions": predictions,
                "total_time": time.time() - start_time
            }
        
        # 获取最主要的预测
        top_prediction = predictions[0]
        mineral_name = top_prediction["mineral_name"]
        confidence = top_prediction["confidence"]
        
        # 2. 查询数据库
        mineral_info = self.query_database(mineral_name)
        
        if mineral_info is None:
            return {
                "success": False,
                "error": f"数据库中未找到矿物：{mineral_name}",
                "predictions": predictions,
                "total_time": time.time() - start_time
            }
        
        # 3. 生成回答
        generation_start = time.time()
        answer = self.generate_answer(mineral_name, mineral_info, user_question, temperature, max_tokens)
        generation_time = time.time() - generation_start
        
        # 返回完整结果
        return {
            "success": True,
            "identification": {
                "mineral_name": mineral_name,
                "confidence": confidence,
                "all_predictions": predictions
            },
            "mineral_info": mineral_info,
            "answer": answer,
            "generation_time": generation_time,
            "total_time": time.time() - start_time
        }


# ==================== Streamlit Web 界面 ====================

st.set_page_config(
    page_title="矿物图像问答系统",
    page_icon="💎",
    layout="wide"
)

st.title("💎 矿物图像问答系统")
st.markdown("### 基于多模态识别的矿物学智能问答系统")

with st.sidebar:
    st.header("⚙️ 配置")

    st.subheader("模型路径")
    swin_path = st.text_input("Swin 模型", value="/root/rag/models/checkpoints/swin_transformer/best_model_acc7886_ep10.pth")
    swin_id2label_path = st.text_input("Swin 标签映射", value="/root/V2_0_Mineral_QA/model/id2label.json")
    llm_path = st.text_input("LLM 模型", value="/root/V2_0_Mineral_QA/model/Qwen2.5-7B-Instruct")
    database_path = st.text_input("数据库路径", value="/root/V2_0_Mineral_QA/text_data/45.json")

    st.subheader("生成参数")
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7)
    max_tokens = st.slider("Max Tokens", 256, 4096, 1024)

    st.subheader("运行配置")
    device = st.selectbox("设备", ["cuda", "cpu"], index=0)

    if st.button("🔄 重新加载系统"):
        if 'qa_system' in st.session_state:
            del st.session_state.qa_system
        st.rerun()

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📷 上传矿物图片")
    uploaded_file = st.file_uploader(
        "选择矿物图片",
        type=['jpg', 'jpeg', 'png', 'bmp']
    )

    if uploaded_file:
        st.image(uploaded_file, caption="上传的矿物图片", use_column_width=True)

with col2:
    st.subheader("💬 提问")
    query = st.text_input(
        "请输入您的问题",
        placeholder="例如：请介绍一下这种矿物的特征和用途",
        value="请介绍一下这种矿物的特征和用途"
    )

    if st.button("🔍 提交问题", type="primary"):
        if not uploaded_file:
            st.warning("请先上传矿物图片")
        elif not query:
            st.warning("请输入问题")
        else:
            with st.spinner("正在处理..."):
                try:
                    # 初始化系统（如果还没有初始化）
                    if 'qa_system' not in st.session_state:
                        st.session_state.qa_system = SimpleMineralQASystem(
                            swin_model_path=swin_path,
                            swin_id2label_path=swin_id2label_path,
                            llm_model_path=llm_path,
                            database_path=database_path,
                            device=device
                        )

                    # 处理图片
                    image = Image.open(uploaded_file)
                    result = st.session_state.qa_system.process_image(
                        image=image,
                        user_question=query,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )

                    if result["success"]:
                        st.success("✅ 处理完成!")

                        # 识别结果
                        st.subheader("🔍 识别结果")
                        identification = result["identification"]
                        st.write(f"**矿物名称：** {identification['mineral_name']}")
                        st.write(f"**置信度：** {identification['confidence']:.2%}")

                        with st.expander("查看 Top-3 预测"):
                            for pred in identification["all_predictions"]:
                                st.write(f"- {pred['mineral_name']}: {pred['confidence']:.2%}")

                        # 矿物信息
                        st.subheader("📚 矿物详细信息")
                        mineral_info = result["mineral_info"]
                        st.write(f"**中文名称：** {mineral_info.get('name', '')}")
                        st.write(f"**英文名称：** {mineral_info.get('english_name', '')}")

                        with st.expander("查看详细描述"):
                            st.write(mineral_info.get('text', '暂无描述'))

                        # AI 解答
                        st.subheader("🤖 AI 解答")
                        st.write(result["answer"])

                        # 处理时间
                        with st.expander("⏱️ 处理时间"):
                            st.write(f"- 识别与检索: {result['total_time'] - result['generation_time']:.2f}s")
                            st.write(f"- 回答生成: {result['generation_time']:.2f}s")
                            st.write(f"- 总计: {result['total_time']:.2f}s")

                    else:
                        st.error(f"❌ 处理失败: {result.get('error', '未知错误')}")

                        if "predictions" in result:
                            with st.expander("查看识别结果"):
                                for pred in result["predictions"]:
                                    st.write(f"- {pred['mineral_name']}: {pred['confidence']:.2%}")

                except Exception as e:
                    st.error(f"❌ 处理失败: {str(e)}")
                    import traceback
                    with st.expander("查看错误详情"):
                        st.code(traceback.format_exc())

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray; font-size: 12px'>"
    "矿物图像问答系统 | 基于 Swin Transformer + Qwen2.5"
    "</p>",
    unsafe_allow_html=True
)