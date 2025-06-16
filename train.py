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
    accelerator = Accelerator()
    device = accelerator.device

    train_dataset = load_train_dataset()
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    vocab_size = tokenizer.vocab_size
    model = CLIP(vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    for epoch in range(10):
        model.train()
        total_loss = torch.tensor(0.0, device=device)
        total_samples = 0

        progress_bar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}", dynamic_ncols=True)

        for images, texts in train_loader:
            optimizer.zero_grad()
            logits_per_image, logits_per_text = model(images, texts)
            loss = clip_loss(logits_per_image, logits_per_text)

            accelerator.backward(loss)
            optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.detach() * batch_size
            total_samples += batch_size

            current_loss = (total_loss / total_samples).item()
            progress_bar.set_postfix({"loss": f"{current_loss:.4f}"})
            progress_bar.update(1)

        progress_bar.close()

        total_loss = accelerator.reduce(total_loss, reduction="sum")
        total_samples = accelerator.reduce(torch.tensor(total_samples, device=device), reduction="sum")
        avg_loss = (total_loss / total_samples).item()

        if accelerator.is_main_process:
            logger.info(f"Epoch {epoch+1} train loss: {avg_loss:.4f}")

            os.makedirs("checkpoints", exist_ok=True)
            ckpt_path = f"checkpoints/clip_epoch_{epoch+1}.pt"
            accelerator.save(accelerator.get_state_dict(model), ckpt_path)
            logger.info(f"Checkpoint saved at {ckpt_path}")

if __name__ == "__main__":
    train()
