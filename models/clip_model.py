import torch
import torch.nn as nn
from encoders.vision_encoder import VisionTransformerEncoder
from encoders.text_encoder import TextTransformerEncoder

class CLIP(nn.Module):
    def __init__(self, vocab_size, embed_dim=512):
        super().__init__()
        self.image_encoder = VisionTransformerEncoder(embed_dim)
        self.text_encoder = TextTransformerEncoder(vocab_size, embed_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1/0.07)))

    def forward(self, images, texts):
        image_features = self.image_encoder(images)   # (batch, embed_dim)
        text_features = self.text_encoder(texts)      # (batch, embed_dim)
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text
