"""
Swin Transformer 矿物图像分类微调训练脚本
支持带标签平滑的交叉熵损失函数、AdamW 优化器、以及多 GPU 分布式训练
"""
import os
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    SwinForImageClassification,
    SwinConfig,
    get_scheduler,
    AutoImageProcessor
)
from PIL import Image
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import random


# ==================== 标签平滑交叉熵损失 ====================

class LabelSmoothingCrossEntropy(nn.Module):
    """
    带标签平滑的交叉熵损失函数
    标签平滑可以防止模型过度自信，提高泛化能力
    
    公式: loss = (1 - smoothing) * CE(y, pred) + smoothing * mean(CE(all, pred))
    """
    
    def __init__(self, smoothing: float = 0.1, reduction: str = 'mean'):
        """
        初始化标签平滑损失
        
        Args:
            smoothing: 平滑因子，取值范围 [0, 1]，值越大平滑程度越高
            reduction: 损失归约方式 ('mean', 'sum', 'none')
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            pred: 模型预测 logits, shape: [batch_size, num_classes]
            target: 真实标签, shape: [batch_size]
        
        Returns:
            损失值
        """
        num_classes = pred.size(-1)
        
        # 将 target 转换为 one-hot 编码
        log_probs = F.log_softmax(pred, dim=-1)
        
        # 创建平滑标签
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        
        # 计算 KL 散度
        loss = torch.sum(-true_dist * log_probs, dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class LabelSmoothingLoss(nn.Module):
    """
    另一种标签平滑实现（基于概率分布）
    """
    
    def __init__(self, num_classes: int, smoothing: float = 0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: predictions, shape: [batch, num_classes]
            target: targets, shape: [batch]
        """
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            # 创建平滑标签
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))


# ==================== 矿物图像数据集 ====================

class MineralDataset(Dataset):
    """
    矿物图像数据集
    支持从文件夹自动加载或从 JSON/CSV 文件加载
    """
    
    def __init__(
        self,
        image_dir: str = None,
        annotations_file: str = None,
        label2id: Dict[str, int] = None,
        image_size: int = 224,
        is_training: bool = True,
        image_processor: AutoImageProcessor = None
    ):
        """
        初始化数据集
        
        Args:
            image_dir: 图像目录路径
            annotations_file: 标注文件路径 (JSON/CSV)
            label2id: 标签到ID的映射字典
            image_size: 图像尺寸
            is_training: 是否为训练模式（会应用数据增强）
            image_processor: HuggingFace 图像预处理器
        """
        self.image_dir = Path(image_dir) if image_dir else None
        self.image_size = image_size
        self.is_training = is_training
        self.image_processor = image_processor
        
        # 加载标注数据
        self.samples = self._load_annotations(annotations_file)
        
        # 构建标签映射
        if label2id is None:
            self.label2id = self._build_label_mapping()
        else:
            self.label2id = label2id
        
        self.id2label = {v: k for k, v in self.label2id.items()}
        
        # 数据增强
        if is_training:
            from torchvision import transforms
            self.transform = transforms.Compose([
                transforms.Resize((image_size + 32, image_size + 32)),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            from torchvision import transforms
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def _load_annotations(self, annotations_file: str) -> List[Tuple[str, int]]:
        """
        加载标注数据
        
        Args:
            annotations_file: 标注文件路径
        
        Returns:
            List of (image_path, label_id) tuples
        """
        samples = []
        
        if annotations_file is None:
            # 从文件夹自动加载（子文件夹名为标签名）
            if self.image_dir and self.image_dir.exists():
                for label_dir in sorted(self.image_dir.iterdir()):
                    if label_dir.is_dir():
                        label_name = label_dir.name
                        for img_file in label_dir.iterdir():
                            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                                samples.append((str(img_file), label_name))
        else:
            # 从标注文件加载
            annotations_file = Path(annotations_file)
            if annotations_file.suffix == '.json':
                with open(annotations_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for item in data:
                    samples.append((item['image_path'], item['label']))
            elif annotations_file.suffix == '.csv':
                import pandas as pd
                df = pd.read_csv(annotations_file)
                for _, row in df.iterrows():
                    samples.append((row['image_path'], row['label']))
        
        return samples
    
    def _build_label_mapping(self) -> Dict[str, int]:
        """从样本中构建标签映射"""
        unique_labels = sorted(set([label for _, label in self.samples]))
        return {label: idx for idx, label in enumerate(unique_labels)}
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        获取单个样本
        
        Returns:
            包含 pixel_values 和 labels 的字典
        """
        image_path, label = self.samples[idx]
        
        # 加载图像
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Failed to load image {image_path}: {e}")
            # 返回空白图像
            image = Image.new('RGB', (self.image_size, self.image_size), (0, 0, 0))
        
        # 转换图像
        if self.image_processor:
            # 使用 HuggingFace processor
            inputs = self.image_processor(images=image, return_tensors="pt")
            pixel_values = inputs['pixel_values'].squeeze(0)
        else:
            # 使用 torchvision transform
            pixel_values = self.transform(image)
        
        # 获取标签 ID
        label_id = self.label2id.get(label, 0)
        
        return {
            'pixel_values': pixel_values,
            'labels': label_id,
            'image_path': image_path,
            'label_name': label
        }


# ==================== 训练器 ====================

class SwinTrainer:
    """
    Swin Transformer 训练器
    支持标签平滑、混合精度训练、梯度累积、学习率调度
    """
    
    def __init__(
        self,
        model: SwinForImageClassification,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        label_smoothing: float = 0.1,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        batch_size: int = 16,
        num_epochs: int = 10,
        warmup_steps: int = 500,
        gradient_accumulation_steps: int = 1,
        use_amp: bool = True,
        device: str = "cuda",
        output_dir: str = "./checkpoints"
    ):
        """
        初始化训练器
        
        Args:
            model: Swin 模型
            train_dataset: 训练数据集
            val_dataset: 验证数据集
            label_smoothing: 标签平滑因子
            learning_rate: 学习率
            weight_decay: 权重衰减
            batch_size: 批次大小
            num_epochs: 训练轮数
            warmup_steps: 预热步数
            gradient_accumulation_steps: 梯度累积步数
            use_amp: 是否使用混合精度训练
            device: 运行设备
            output_dir: 输出目录
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_amp = use_amp
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
        
        # 使用标签平滑交叉熵损失
        num_labels = model.config.num_labels
        self.criterion = LabelSmoothingCrossEntropy(
            smoothing=label_smoothing,
            reduction='mean'
        )
        
        # 使用 AdamW 优化器
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # 学习率调度器
        num_training_steps = len(self.train_loader) * num_epochs
        self.scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # 混合精度训练
        self.scaler = GradScaler() if use_amp else None
        
        # 训练状态
        self.global_step = 0
        self.best_val_acc = 0.0
    
    def train(self):
        """执行完整训练流程"""
        print("=" * 60)
        print("开始训练 Swin Transformer 矿物分类模型")
        print("=" * 60)
        
        for epoch in range(self.num_epochs):
            print(f"\n📖 Epoch {epoch + 1}/{self.num_epochs}")
            
            # 训练阶段
            train_loss, train_acc = self.train_epoch(epoch)
            
            # 验证阶段
            if self.val_dataset:
                val_loss, val_acc = self.evaluate(epoch)
                
                print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
                print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
                
                # 保存最佳模型
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.save_checkpoint("best_model.pth", {
                        'epoch': epoch,
                        'val_acc': val_acc
                    })
            else:
                print(f"  Train Loss: {train_loss} | Train Acc: {train_acc}")
            
            # 保存 epoch checkpoint
            self.save_checkpoint(f"epoch_{epoch}.pth", {
                'epoch': epoch,
                'train_acc': train_acc
            })
        
        print("\n✅ 训练完成!")
        print(f"最佳验证准确率: {self.best_val_acc:.4f}")
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # 获取数据
            pixel_values = batch['pixel_values'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # 混合精度训练
            if self.use_amp:
                with autocast():
                    outputs = self.model(pixel_values=pixel_values)
                    logits = outputs.logits
                    loss = self.criterion(logits, labels)
                    loss = loss / self.gradient_accumulation_steps
                
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
            else:
                outputs = self.model(pixel_values=pixel_values)
                logits = outputs.logits
                loss = self.criterion(logits, labels)
                loss = loss / self.gradient_accumulation_steps
                
                loss.backward()
                
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
            
            # 统计
            total_loss += loss.item() * self.gradient_accumulation_steps
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f"{total_loss / (batch_idx + 1):.4f}",
                'acc': f"{100. * correct / total:.2f}%",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            self.global_step += 1
        
        return total_loss / len(self.train_loader), correct / total
    
    @torch.no_grad()
    def evaluate(self, epoch: int = 0) -> Tuple[float, float]:
        """验证模型"""
        if not hasattr(self, 'val_loader'):
            return 0.0, 0.0
        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch in tqdm(self.val_loader, desc="Evaluating"):
            pixel_values = batch['pixel_values'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            outputs = self.model(pixel_values=pixel_values)
            logits = outputs.logits
            loss = self.criterion(logits, labels)
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        return total_loss / len(self.val_loader), correct / total
    
    def save_checkpoint(self, filename: str, metadata: dict = None):
        """保存模型 checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_val_acc': self.best_val_acc,
        }
        
        if metadata:
            checkpoint.update(metadata)
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        save_path = self.output_dir / filename
        torch.save(checkpoint, save_path)
        
        # 保存标签映射
        if hasattr(self.train_dataset, 'label2id'):
            label_path = self.output_dir / "label2id.json"
            with open(label_path, 'w', encoding='utf-8') as f:
                json.dump(self.train_dataset.label2id, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Checkpoint 已保存: {save_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载模型 checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"✅ Checkpoint 已加载: {checkpoint_path}")


# ==================== 主训练函数 ====================

def train_swin_mineral_classifier(args):
    """
    主训练函数
    
    Args:
        args: 命令行参数
    """
    # 设置随机种子
    def set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    set_seed(args.seed)
    
    # 设备配置
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"🖥️  使用设备: {device}")
    
    # 加载模型
    print(f"📦 加载模型: {args.model_name}")
    if args.pretrained:
        model = SwinForImageClassification.from_pretrained(
            args.model_name,
            num_labels=args.num_classes,
            trust_remote_code=True
        )
    else:
        config = SwinConfig(
            image_size=args.image_size,
            patch_size=4,
            num_channels=3,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            num_classes=args.num_classes
        )
        model = SwinForImageClassification(config)
    
    model.to(device)
    
    # 加载数据集
    print(f"📂 加载数据集...")
    image_processor = AutoImageProcessor.from_pretrained(args.model_name)
    
    train_dataset = MineralDataset(
        image_dir=args.train_image_dir,
        annotations_file=args.train_annotations,
        image_size=args.image_size,
        is_training=True,
        image_processor=image_processor
    )
    
    val_dataset = None
    if args.val_image_dir or args.val_annotations:
        val_dataset = MineralDataset(
            image_dir=args.val_image_dir,
            annotations_file=args.val_annotations,
            label2id=train_dataset.label2id,
            image_size=args.image_size,
            is_training=False,
            image_processor=image_processor
        )
    
    print(f"   训练集: {len(train_dataset)} 样本")
    if val_dataset:
        print(f"   验证集: {len(val_dataset)} 样本")
    print(f"   类别数: {len(train_dataset.label2id)}")
    
    # 创建训练器
    trainer = SwinTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        label_smoothing=args.label_smoothing,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_amp=args.use_amp,
        device=str(device),
        output_dir=args.output_dir
    )
    
    # 开始训练
    trainer.train()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Swin Transformer 矿物分类模型训练")
    
    # 模型参数
    parser.add_argument("--model_name", type=str, default="microsoft/swin-tiny-patch4-window7-224",
                       help="预训练模型名称或本地路径")
    parser.add_argument("--num_classes", type=int, default=100,
                       help="分类类别数")
    parser.add_argument("--pretrained", type=bool, default=True,
                       help="是否加载预训练权重")
    parser.add_argument("--image_size", type=int, default=224,
                       help="输入图像尺寸")
    
    # 数据参数
    parser.add_argument("--train_image_dir", type=str, default=None,
                       help="训练图像目录")
    parser.add_argument("--train_annotations", type=str, default=None,
                       help="训练标注文件 (JSON/CSV)")
    parser.add_argument("--val_image_dir", type=str, default=None,
                       help="验证图像目录")
    parser.add_argument("--val_annotations", type=str, default=None,
                       help="验证标注文件")
    
    # 训练参数
    parser.add_argument("--num_epochs", type=int, default=10,
                       help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="权重衰减")
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                       help="标签平滑因子")
    parser.add_argument("--warmup_steps", type=int, default=500,
                       help="预热步数")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="梯度累积步数")
    parser.add_argument("--use_amp", type=bool, default=True,
                       help="是否使用混合精度训练")
    
    # 其他参数
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/swin_mineral",
                       help="输出目录")
    parser.add_argument("--device", type=str, default="cuda",
                       help="设备")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_swin_mineral_classifier(args)
