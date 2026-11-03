"""
Swin Transformer 矿物图像分类推理模型
用于从矿物图像中提取视觉语义锚点，输出 Top-3 矿物类别及置信度
"""
import os
import torch
import torch.nn as nn
from transformers import SwinForImageClassification, SwinConfig
from PIL import Image
import numpy as np
from typing import List, Tuple, Dict, Optional
import json
from pathlib import Path


class MineralSwinClassifier:
    """
    Swin Transformer 矿物图像分类器
    支持加载预训练权重并进行推理，输出 Top-K 预测结果及置信度
    """

    def __init__(
        self,
        model_path: str,
        id2label_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        top_k: int = 3
    ):
        """
        初始化 Swin 分类器

        Args:
            model_path: 模型权重路径 (本地 .pth 文件或 HuggingFace Hub)
            id2label_path: 标签映射文件路径 (JSON格式)
            device: 运行设备 ("cuda" 或 "cpu")
            top_k: 输出 Top-K 预测结果
        """
        self.device = device
        self.top_k = top_k
        self.model_path = model_path

        # 判断是 .pth 文件还是 HuggingFace 模型
        if model_path.endswith('.pth') or model_path.endswith('.pt'):
            # 本地 PyTorch checkpoint
            self._load_from_checkpoint(model_path)
        else:
            # HuggingFace 模型
            self._load_from_huggingface(model_path)

        # 加载标签映射
        if id2label_path and Path(id2label_path).exists():
            with open(id2label_path, 'r', encoding='utf-8') as f:
                self.id2label = json.load(f)
        else:
            # 使用模型默认的标签映射
            self.id2label = self.config.id2label if hasattr(self.config, 'id2label') else None

        print(f"✅ Swin 模型加载成功: {model_path}")
        print(f"   设备: {self.device}")
        print(f"   类别数: {self.config.num_labels}")

    def _load_from_checkpoint(self, checkpoint_path: str):
        """从 PyTorch checkpoint 加载模型 (mmpretrain 格式)"""
        print(f"   从 mmpretrain checkpoint 加载: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # 使用 timm 创建 backbone (num_classes=0 输出特征向量)
        import timm
        self.backbone = timm.create_model(
            'swin_base_patch4_window7_224_in22k',
            pretrained=False,
            num_classes=0,  # 输出特征向量
            drop_path_rate=0.1,
            img_size=224
        )

        # 构建与训练时完全一致的分类头
        embed_dim = self.backbone.num_features  # 1024 for swin_base
        num_classes = 45

        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.1),
            torch.nn.Linear(embed_dim, embed_dim // 2),  # 1024 -> 512
            torch.nn.GELU(),
            torch.nn.Dropout(0.05),
            torch.nn.Linear(embed_dim // 2, num_classes)  # 512 -> 45
        )

        # 重新组装 state_dict，将 classifier 权重映射到 classifier
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key

            # backbone.patch_embed -> patch_embed
            # backbone.layers -> layers
            new_key = new_key.replace('backbone.', '')

            # 注意：classifier.1 和 classifier.4 保持不变，因为 Sequential 的索引就是 1 和 4

            if new_key != key:
                print(f"   Remap: {key} -> {new_key}")
            new_state_dict[new_key] = value

        # 加载 backbone 权重
        backbone_state_dict = {}
        classifier_state_dict = {}
        for k, v in new_state_dict.items():
            if k.startswith('patch_embed.') or k.startswith('layers.') or k.startswith('head.') or k.startswith('norm.'):
                backbone_state_dict[k] = v
            elif k.startswith('classifier.'):
                # 需要去掉 classifier. 前缀，Sequential 期望的是 1.weight 而不是 classifier.1.weight
                new_k = k.replace('classifier.', '')
                classifier_state_dict[new_k] = v

        print(f"   Backbone keys: {len(backbone_state_dict)}, Classifier keys: {len(classifier_state_dict)}")

        self.backbone.load_state_dict(backbone_state_dict, strict=True)
        self.backbone.to(self.device)
        self.backbone.eval()

        self.classifier.load_state_dict(classifier_state_dict, strict=False)
        self.classifier.to(self.device)
        self.classifier.eval()

        # 创建完整模型封装
        class ModelWithClassifier(torch.nn.Module):
            def __init__(self, backbone, classifier):
                super().__init__()
                self.backbone = backbone
                self.classifier = classifier

            def forward(self, x):
                features = self.backbone(x)
                return self.classifier(features)

        self.model = ModelWithClassifier(self.backbone, self.classifier)
        self.model.eval()

        # 为模型创建简单的 config 模拟对象
        class ModelConfig:
            num_labels = 45
            image_size = 224
        self.config = ModelConfig()

        # 使用 timm 的数据配置进行预处理
        self.timm_config = self.backbone.default_cfg
        self.data_config = timm.data.resolve_data_config(self.timm_config)
        self.transform = timm.data.create_transform(**self.data_config, is_training=False)

        print(f"   成功加载 {len(state_dict)} 个权重")
        print(f"   模型类型: Swin Base 224 + 2-layer classifier, num_classes=45")
        print(f"   Embed dim: {embed_dim}")

    def _load_from_huggingface(self, model_path: str):
        """从 HuggingFace 加载模型"""
        self.model = SwinForImageClassification.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        self.model.to(self.device)
        self.model.eval()
        self.config = self.model.config
    
    @torch.no_grad()
    def predict(
        self,
        image: Image.Image,
        return_probs: bool = True
    ) -> List[Tuple[str, float]]:
        """
        对输入图像进行分类预测

        Args:
            image: PIL 图像对象
            return_probs: 是否返回概率值

        Returns:
            List of (label, confidence) tuples, sorted by confidence descending
        """
        # 图像预处理
        inputs = self._preprocess_image(image)

        # 前向传播
        if isinstance(inputs, dict) and 'pixel_values' in inputs:
            # 使用封装的模型
            outputs = self.model(inputs['pixel_values'])
        else:
            outputs = self.model(**inputs)

        # 获取 logits - 模型直接返回 tensor
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs  # 直接返回 tensor

        print(f"   [DEBUG] logits shape: {logits.shape}, max: {logits.max():.4f}, min: {logits.min():.4f}")

        probs = torch.softmax(logits, dim=-1)

        # 获取 Top-K 预测
        top_probs, top_indices = torch.topk(probs, self.top_k, dim=-1)

        print(f"   [DEBUG] top_probs: {top_probs}, top_indices: {top_indices}")

        # 转换为 Python 列表
        top_probs = top_probs.squeeze().cpu().tolist()
        top_indices = top_indices.squeeze().cpu().tolist()
        
        # 构建结果列表
        results = []
        for idx, prob in zip(top_indices, top_probs):
            if self.id2label:
                label = self.id2label.get(str(idx), self.id2label.get(idx, f"class_{idx}"))
            else:
                label = f"class_{idx}"
            results.append((label, prob))
        
        return results
    
    @torch.no_grad()
    def predict_with_features(
        self,
        image: Image.Image
    ) -> Dict:
        """
        带详细特征的预测结果
        
        Args:
            image: PIL 图像对象
        
        Returns:
            包含预测结果、概率、特征向量等完整信息
        """
        # 图像预处理
        inputs = self._preprocess_image(image)
        
        # 获取图像特征
        outputs = self.model(**inputs, output_hidden_states=True)
        
        # 获取 logits 和隐藏层
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        
        # 获取最后一个隐藏层作为图像特征
        hidden_states = outputs.hidden_states
        image_features = hidden_states[-1][:, 0, :].squeeze().cpu().numpy()  # [CLS] token
        
        # Top-K 预测
        top_probs, top_indices = torch.topk(probs, self.top_k, dim=-1)
        
        predictions = []
        for idx, prob in zip(top_indices.squeeze(), top_probs.squeeze()):
            if self.id2label:
                label = self.id2label.get(str(idx.item()), self.id2label.get(idx.item(), f"class_{idx}"))
            else:
                label = f"class_{idx.item()}"
            predictions.append({
                "label": label,
                "confidence": prob.item(),
                "class_id": idx.item()
            })
        
        return {
            "predictions": predictions,
            "image_features": image_features,
            "logits": logits.squeeze().cpu().tolist()
        }
    
    def _preprocess_image(self, image: Image.Image) -> Dict:
        """
        图像预处理 - 将 PIL 图像转换为模型输入

        Args:
            image: PIL 图像对象

        Returns:
            包含 pixel_values 的字典
        """
        # 确保图像为 RGB 模式
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # 如果使用 timm 模型，使用 timm 的 transform
        if hasattr(self, 'transform'):
            inputs = self.transform(image).unsqueeze(0).to(self.device)
            return {"pixel_values": inputs}

        # 否则使用 transformers processor
        from transformers import AutoImageProcessor
        if not hasattr(self, 'image_processor'):
            try:
                self.image_processor = AutoImageProcessor.from_pretrained(
                    self.model_path if hasattr(self, 'model_path') else "microsoft/swin-tiny-patch4-window7-224"
                )
            except:
                self.image_processor = None

        if self.image_processor:
            inputs = self.image_processor(images=image, return_tensors="pt")
        else:
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
            pixel_values = transform(image).unsqueeze(0)
            inputs = {"pixel_values": pixel_values}

        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(self.device)

        return inputs
    
    def get_visual_anchors(
        self,
        image: Image.Image,
        confidence_threshold: float = 0.01
    ) -> Dict:
        """
        获取视觉语义锚点 - V2.0 核心接口
        
        Args:
            image: PIL 图像对象
            confidence_threshold: 置信度阈值
        
        Returns:
            包含 Top-3 锚点及置信度的字典
        """
        predictions = self.predict(image)

        print(f"   [DEBUG] predictions: {predictions}")

        # 过滤低置信度结果
        filtered_preds = [(label, prob) for label, prob in predictions if prob >= confidence_threshold]
        
        # 返回 Top-3 锚点
        anchors = []
        for i, (label, prob) in enumerate(filtered_preds[:3]):
            anchors.append({
                "rank": i + 1,
                "mineral_name": label,
                "confidence": prob,
                "is_primary": i == 0  # 标记是否为最主要的预测
            })
        
        return {
            "visual_anchors": anchors,
            "total_predictions": len(predictions),
            "image_processed": True
        }


class SwinModelWrapper:
    """
    Swin Transformer 模型封装类
    提供更灵活的接口，支持自定义前向传播逻辑
    """
    
    def __init__(
        self,
        model_path: str,
        num_labels: int = None,
        pretrained: bool = True,
        device: str = "cuda"
    ):
        """
        初始化 Swin 模型封装
        
        Args:
            model_path: 模型路径或 HuggingFace 模型 ID
            num_labels: 分类类别数 (用于从头训练)
            pretrained: 是否加载预训练权重
            device: 运行设备
        """
        self.device = device
        
        if pretrained:
            # 加载预训练模型
            self.model = SwinForImageClassification.from_pretrained(
                model_path,
                trust_remote_code=True
            )
        else:
            # 从配置创建模型
            if num_labels is None:
                raise ValueError("从头训练时必须指定 num_labels")
            
            config = SwinConfig(
                image_size=224,
                patch_size=4,
                num_channels=3,
                embed_dim=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                num_classes=num_labels
            )
            self.model = SwinForImageClassification(config)
        
        self.model.to(self.device)
        self.config = self.model.config
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        自定义前向传播逻辑
        
        Args:
            pixel_values: 图像像素值 tensor, shape: [batch, channels, height, width]
        
        Returns:
            分类 logits
        """
        pixel_values = pixel_values.to(self.device)
        outputs = self.model(pixel_values=pixel_values)
        return outputs.logits
    
    def extract_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        提取图像特征向量
        
        Args:
            pixel_values: 图像像素值 tensor
        
        Returns:
            图像特征向量
        """
        pixel_values = pixel_values.to(self.device)
        outputs = self.model(
            pixel_values=pixel_values,
            output_hidden_states=True
        )
        # 使用 [CLS] token 作为图像表示
        features = outputs.hidden_states[-1][:, 0, :]
        return features
    
    def predict_top_k(
        self,
        pixel_values: torch.Tensor,
        top_k: int = 3
    ) -> Tuple[List[int], List[float]]:
        """
        预测 Top-K 结果
        
        Args:
            pixel_values: 图像像素值
            top_k: 返回 Top-K 结果
        
        Returns:
            (类别ID列表, 概率列表)
        """
        logits = self.forward(pixel_values)
        probs = torch.softmax(logits, dim=-1)
        
        top_probs, top_indices = torch.topk(probs, top_k, dim=-1)
        
        return top_indices.squeeze().tolist(), top_probs.squeeze().tolist()
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        加载自定义训练 checkpoint
        
        Args:
            checkpoint_path: checkpoint 文件路径
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        print(f"✅ Checkpoint 加载成功: {checkpoint_path}")
    
    def save_checkpoint(self, checkpoint_path: str, **kwargs):
        """
        保存模型 checkpoint
        
        Args:
            checkpoint_path: 保存路径
            **kwargs: 其他需要保存的信息 (如 epoch, optimizer 等)
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            **kwargs
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"✅ Checkpoint 保存成功: {checkpoint_path}")


def load_swin_for_inference(
    model_path: str,
    id2label_path: str = None,
    device: str = "cuda"
) -> MineralSwinClassifier:
    """
    工厂函数：加载 Swin 模型用于推理
    
    Args:
        model_path: 模型路径
        id2label_path: 标签映射文件
        device: 运行设备
    
    Returns:
        MineralSwinClassifier 实例
    """
    return MineralSwinClassifier(
        model_path=model_path,
        id2label_path=id2label_path,
        device=device
    )


# ============ 示例用法 ============
if __name__ == "__main__":
    import os
    
    # 示例配置
    MODEL_PATH = "microsoft/swin-tiny-patch4-window7-224"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 50)
    print("Swin Transformer 矿物分类器示例")
    print("=" * 50)
    
    # 方式1: 使用工厂函数加载
    classifier = load_swin_for_inference(
        model_path=MODEL_PATH,
        device=DEVICE
    )
    
    # 方式2: 直接使用封装类
    swin_wrapper = SwinModelWrapper(
        model_path=MODEL_PATH,
        pretrained=True,
        device=DEVICE
    )
    
    print("\n✅ 模型加载完成，可以开始推理")
    
    # 注意: 实际使用时需要准备矿物图像数据集
    # 示例: image = Image.open("mineral_sample.jpg")
    # predictions = classifier.predict(image)
    # print(f"预测结果: {predictions}")
