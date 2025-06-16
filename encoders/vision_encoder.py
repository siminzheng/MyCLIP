import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16

class VisionTransformerEncoder(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        # 加载ViT base模型（非预训练）
        self.vit = vit_b_16(pretrained=False)
        # 去掉分类头
        self.vit.heads = nn.Identity()
        # 线性投影，将768维向量映射到embed_dim维
        self.proj = nn.Linear(768, embed_dim)

    def forward(self, x):
        x = self.vit(x)              # 输出shape: (batch_size, 768)
        x = self.proj(x)             # 映射到 embed_dim
        return F.normalize(x, dim=-1) # L2归一化
