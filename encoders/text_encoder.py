import torch
import torch.nn as nn
import torch.nn.functional as F

class TextTransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, num_layers=6, num_heads=8, max_len=32):
        super().__init__()
        # 词嵌入层，将token id映射为向量，维度为embed_dim
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        # 位置嵌入参数，形状(1, max_len, embed_dim)，用于编码序列中每个位置的信息
        self.position_embedding = nn.Parameter(torch.randn(1, max_len, embed_dim))
        # Transformer编码器层，包含多头自注意力机制和前馈网络
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        # 堆叠num_layers个Transformer编码器层
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 最后的LayerNorm层，归一化输出
        self.ln_final = nn.LayerNorm(embed_dim)
        self.max_len = max_len

    def forward(self, x):
        # x的形状是(batch_size, seq_len)，是token的索引序列
        token_embeds = self.token_embedding(x)  
        # token_embeds形状：(batch_size, seq_len, embed_dim)

        # 取对应序列长度的位置信息嵌入
        position_embeds = self.position_embedding[:, :x.size(1), :]
        # 把词向量和位置向量相加，融合位置信息
        x = token_embeds + position_embeds

        # 传入Transformer编码器，捕捉序列内部的上下文关系
        x = self.transformer(x)  # (batch_size, seq_len, embed_dim)

        # 经过LayerNorm归一化
        x = self.ln_final(x)

        # 对序列维度求均值，得到每个样本的固定长度句子向量表示
        x = x.mean(dim=1)  # (batch_size, embed_dim)

        # 对句子向量做L2归一化，方便后续计算相似度
        return F.normalize(x, dim=-1)
