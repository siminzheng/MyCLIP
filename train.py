import os
import torch
import logging
from tqdm import tqdm
from accelerate import Accelerator
from torch.utils.data import DataLoader
from data.dataset import load_train_dataset
from models.clip_model import CLIP
from utils.loss_utils import clip_loss
from utils.tokenizer_utils import tokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train():
    # 初始化Accelerator，简化多设备（GPU/TPU）训练流程
    accelerator = Accelerator()
    device = accelerator.device  # 获取当前设备（CPU或GPU）

    # 加载训练数据集
    train_dataset = load_train_dataset()
    # 用DataLoader包装数据集，batch_size=32，多线程加载数据
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    # 获取词表大小，构建模型时需要
    vocab_size = tokenizer.vocab_size
    # 初始化CLIP模型，移动到当前设备
    model = CLIP(vocab_size).to(device)
    # AdamW优化器，学习率1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # 用accelerator.prepare包装模型、优化器和数据加载器，支持分布式训练、混合精度等
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    for epoch in range(10):
        model.train()  # 进入训练模式
        total_loss = torch.tensor(0.0, device=device)  # 累计损失
        total_samples = 0  # 累计样本数

        # tqdm进度条显示当前epoch的训练进度
        progress_bar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}", dynamic_ncols=True)

        for images, texts in train_loader:
            optimizer.zero_grad()  # 清空梯度
            logits_per_image, logits_per_text = model(images, texts)  # 前向传播，得到图像和文本的logits
            loss = clip_loss(logits_per_image, logits_per_text)  # 计算CLIP特定的对比损失

            accelerator.backward(loss)  # 反向传播，支持混合精度
            optimizer.step()  # 参数更新

            batch_size = images.size(0)  # 当前batch大小
            # 累计加权损失，用于计算平均损失
            total_loss += loss.detach() * batch_size
            total_samples += batch_size

            current_loss = (total_loss / total_samples).item()
            # 更新进度条右侧显示当前平均loss
            progress_bar.set_postfix({"loss": f"{current_loss:.4f}"})
            progress_bar.update(1)

        progress_bar.close()

        # 跨多设备汇总损失和样本数
        total_loss = accelerator.reduce(total_loss, reduction="sum")
        total_samples = accelerator.reduce(torch.tensor(total_samples, device=device), reduction="sum")
        avg_loss = (total_loss / total_samples).item()

        # 只有主进程打印日志和保存模型，避免重复操作
        if accelerator.is_main_process:
            logger.info(f"Epoch {epoch+1} train loss: {avg_loss:.4f}")

            # 创建保存模型的目录
            os.makedirs("checkpoints", exist_ok=True)
            ckpt_path = f"checkpoints/clip_epoch_{epoch+1}.pt"
            # 保存当前模型状态字典（支持多设备）
            accelerator.save(accelerator.get_state_dict(model), ckpt_path)
            logger.info(f"Checkpoint saved at {ckpt_path}")

if __name__ == "__main__":
    train()
