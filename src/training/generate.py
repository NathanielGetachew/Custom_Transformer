import torch

def generate_text(model, tokenizer, device, prompt_text, max_new_tokens=200, top_k=50):
    model.eval()
    tokens = torch.tensor([tokenizer.encode_ids(prompt_text)], dtype=torch.long).to(device)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _ = model(tokens)
            logits = logits[:, -1, :]
            top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
            probs = torch.softmax(top_k_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            next_token = top_k_indices.gather(-1, next_token)
            tokens = torch.cat([tokens, next_token], dim=1)
            if next_token.item() == tokenizer.token_to_id["<eos>"]:
                break
    return tokenizer.decode_ids(tokens[0].tolist())