import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader
from src.data.tensor_dataset import TensorDataset
from src.models.transformer import TransformerDecoder
from src.training.train import train_model
from src.utils.plotting import plot_losses
from src.data.tokenizer import CustomTokenizer

config = {
    "batch_size": 4,
    "max_length": 64,
    "d_model": 128,
    "nhead": 4,
    "num_layers": 2,
    "dim_feedforward": 512,
    "dropout": 0.1,
    "learning_rate": 3e-4,
    "max_epochs": 1,
    "eval_interval": 100,
    "grad_clip": 1.0,
    "device": "cpu",
    "ckpt_path": "checkpoints/model_ckpt.pth",
    "final_model_path": "checkpoints/transformer_final.pth"
}

def main():
    train_dataset = TensorDataset("data/processed/train.pt")
    val_dataset = TensorDataset("data/processed/valid.pt")

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0, pin_memory=False)

    tokenizer = CustomTokenizer()

    model = TransformerDecoder(
        vocab_size=tokenizer.vocab_size,
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        dim_feedforward=config["dim_feedforward"],
        max_len=config["max_length"],
        dropout=config["dropout"]
    )

    steps, train_losses, val_losses = train_model(model, train_loader, val_loader, tokenizer, config)
    plot_losses(steps, train_losses, val_losses)

if __name__ == "__main__":
    main()