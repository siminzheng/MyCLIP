import torch
import torch.nn as nn
import torch.nn.functional as F

class TextTransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, num_layers=6, num_heads=8, max_len=32):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Parameter(torch.randn(1, max_len, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln_final = nn.LayerNorm(embed_dim)
        self.max_len = max_len

    def forward(self, x):
        token_embeds = self.token_embedding(x)  # (batch, seq_len, embed_dim)
        position_embeds = self.position_embedding[:, :x.size(1), :]
        x = token_embeds + position_embeds
        x = self.transformer(x)       # Transformer编码
        x = self.ln_final(x)
        x = x.mean(dim=1)             # 取序列平均作为句子向量
        return F.normalize(x, dim=-1) # L2归一化
