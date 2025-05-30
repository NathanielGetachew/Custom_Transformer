# src/training/train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import math
import os

def get_batch(data_loader, device):
    for batch in data_loader:
        inputs, targets = batch
        yield inputs.to(device), targets.to(device)

def train_model(model, train_loader, val_loader, tokenizer, config):
    device = config["device"]
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], betas=(0.9, 0.95), weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["max_epochs"] * len(train_loader), eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id["<pad>"])
    scaler = GradScaler('cuda') if device.startswith("cuda") else None

    train_losses = []
    val_losses = []
    steps = []
    global_step = 0

    for epoch in range(config["max_epochs"]):
        model.train()
        total_loss = 0
        train_iter = get_batch(train_loader, device)
        
        for step, (inputs, targets) in enumerate(tqdm(train_iter, total=len(train_loader), desc=f"Epoch {epoch+1}/{config['max_epochs']}")):
            if device.startswith("cuda"):
                with autocast('cuda'):
                    logits, _ = model(inputs)
                    targets = targets[:, :-1]
                    loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            else:
                logits, _ = model(inputs)
                targets = targets[:, :-1]
                loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            if device.startswith("cuda"):
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                optimizer.step()
            
            optimizer.zero_grad()
            scheduler.step()
            total_loss += loss.item()
            global_step += 1

            if global_step % config["eval_interval"] == 0:
                val_loss = evaluate_loss(model, val_loader, criterion, device)
                print(f"Step {global_step}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")
                train_losses.append(loss.item())
                val_losses.append(val_loss)
                steps.append(global_step)
                save_checkpoint(model, optimizer, config, epoch, global_step)

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Avg Train Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), config["final_model_path"])
    print(f"Final model saved to {config['final_model_path']}")
    return steps, train_losses, val_losses

def evaluate_loss(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in get_batch(val_loader, device):
            logits, _ = model(inputs)
            targets = targets[:, :-1]
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            total_loss += loss.item()
    return total_loss / len(val_loader)

def save_checkpoint(model, optimizer, config, epoch, step):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "step": step
    }
    torch.save(checkpoint, config["ckpt_path"])
    print(f"Checkpoint saved at epoch {epoch+1}, step {step}")