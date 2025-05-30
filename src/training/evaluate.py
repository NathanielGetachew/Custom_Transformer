# src/training/evaluate.py
import torch
import torch.nn as nn

def evaluate_model(model, test_loader, tokenizer, device):
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id["<pad>"])
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits, _ = model(inputs)
            targets = targets[:, :-1]
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            total_loss += loss.item()
    mean_loss = total_loss / len(test_loader)
    perplexity = torch.exp(torch.tensor(mean_loss))
    return mean_loss, perplexity