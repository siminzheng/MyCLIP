# MyCLIP: 轻量级图像-文本对比学习模型

```text


                            ███╗   ███╗██╗   ██╗ ██████╗██╗     ██╗██████╗ 
                            ████╗ ████║╚██╗ ██╔╝██╔════╝██║     ██║██╔══██╗
                            ██╔████╔██║ ╚████╔╝ ██║     ██║     ██║██████╔╝
                            ██║╚██╔╝██║  ╚██╔╝  ██║     ██║     ██║██╔═══╝ 
                            ██║ ╚═╝ ██║   ██║   ╚██████╗███████╗██║██║     
                            ╚═╝     ╚═╝   ╚═╝    ╚═════╝╚══════╝╚═╝╚═╝     
                                       

```             

## 项目简介

本项目实现了一个简化版的 CLIP 模型，结合了 Vision Transformer (ViT) 和 Transformer 文本编码器，支持图像与文本的对比学习。使用 CIFAR-10 数据集进行训练和测试，支持多卡加速和混合精度。  
This project implements a simplified version of the CLIP model, combining a Vision Transformer (ViT) and a Transformer-based text encoder to support image-text contrastive learning. It is trained and tested on the CIFAR-10 dataset and supports multi-GPU acceleration as well as mixed precision.

---

## 项目目录结构

```text

MyCLIP/
├── checkpoints/ # 训练模型保存目录
├── encoders/ # 编码器模块
│ ├── vision_encoder.py
│ └── text_encoder.py
├── models/ # CLIP模型主体
│ └── clip_model.py
├── data/ # 数据集及加载
│ └── dataset.py
├── utils/ # 工具函数
│ ├── tokenizer_utils.py
│ └── loss_utils.py
├── accelerate_config.yaml #配置文件
├── train.py # 训练脚本
├── evaluate.py # 测试脚本
├── requirements.txt # 环境依赖
└── README.md # 项目说明


```

## 环境安装

```bash
pip install -r requirements.txt
```

启动训练
使用 Accelerate 启动，支持多卡和混合精度（需提前配置 accelerate config 或修改命令参数）

```bash
accelerate launch train.py
```
如果需要 DeepSpeed 支持，请先配置 DeepSpeed 配置文件，并用如下命令(默认配置文件 accelerate_config.yaml 中的配置为1个机器，2张卡)：

```bash
accelerate launch --config_file ./configs/deepspeed_config.yaml train.py
```
启动测试
训练完成后，执行：


```bash
python evaluate.py
```
备注
训练使用 CIFAR-10 图像和对应类别名文本对进行对比学习

文本Tokenizer基于 Huggingface BERT Tokenizer

图像编码器基于 torchvision ViT Base 模型（非预训练）

训练过程自动保存模型检查点至 checkpoints/

评估时加载最后一轮模型权重（默认clip_epoch_10.pt）  

欢迎反馈和讨论！  


Notes:

Training uses CIFAR-10 images and their corresponding class name text pairs for contrastive learning.

The text tokenizer is based on the Huggingface BERT Tokenizer.

The image encoder is based on the torchvision ViT-Base model (without pre-training).

Model checkpoints are automatically saved to the checkpoints/ directory during training.

During evaluation, the final model weights (default: clip_epoch_10.pt) are loaded.

Feedback and discussions are warmly welcome!


