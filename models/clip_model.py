import torch
import torch.nn as nn
from encoders.vision_encoder import VisionTransformerEncoder
from encoders.text_encoder import TextTransformerEncoder

class CLIP(nn.Module):
    def __init__(self, vocab_size, embed_dim=512):
        super().__init__()
        # 图像编码器，使用Vision Transformer结构，输出维度为embed_dim
        self.image_encoder = VisionTransformerEncoder(embed_dim)
        # 文本编码器，基于Transformer结构，词汇表大小为vocab_size，输出维度同为embed_dim
        self.text_encoder = TextTransformerEncoder(vocab_size, embed_dim)
        # logit_scale是一个可训练的缩放因子，初始化为log(1/0.07)，用于调节相似度的尺度
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1/0.07)))

    def forward(self, images, texts):
        # 输入images和texts，分别经过对应编码器，得到形状均为(batch_size, embed_dim)的特征向量
        image_features = self.image_encoder(images)   # 图像特征
        text_features = self.text_encoder(texts)      # 文本特征
        
        # 将logit_scale取指数，确保缩放因子为正数
        logit_scale = self.logit_scale.exp()
        
        # 计算图像特征与文本特征的相似度（点积），并乘以缩放因子
        # logits_per_image形状为(batch_size, batch_size)，第i行j列表示第i张图与第j条文本的相似度
        logits_per_image = logit_scale * image_features @ text_features.t()
        
        # logits_per_text是对上面矩阵的转置，方便双向计算
        logits_per_text = logits_per_image.t()
        
        # 返回图像对文本和文本对图像的相似度矩阵，用于后续的损失计算
        return logits_per_image, logits_per_text
