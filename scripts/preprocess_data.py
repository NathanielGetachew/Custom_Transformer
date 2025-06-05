import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import torch
import os
from tqdm import tqdm
from src.data.tokenizer import CustomTokenizer

def preprocess_data(data_dir="data/ptbdataset", output_dir="data/processed", max_length=64):
    os.makedirs(output_dir, exist_ok=True)
    splits = ["train", "valid", "test"]
    tokenizer = CustomTokenizer()

    for split in splits:
        input_path = os.path.join(data_dir, f"ptb.{split}.txt")
        output_path = os.path.join(output_dir, f"{split}.pt")

        with open(input_path, "r", encoding="utf-8") as f:
            lines = [line.strip().lower() for line in f if line.strip()]

        inputs = []
        targets = []

        for line in tqdm(lines, desc=f"Tokenizing {split}"):
            token_ids = tokenizer.encode_ids(line)
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
            input_seq = token_ids[:-1]
            target_seq = token_ids[1:]
            pad_id = tokenizer.token_to_id["<pad>"]
            input_seq = input_seq + [pad_id] * (max_length - 1 - len(input_seq))
            target_seq = target_seq + [pad_id] * (max_length - 1 - len(target_seq))
            inputs.append(input_seq)
            targets.append(target_seq)

        inputs_tensor = torch.tensor(inputs, dtype=torch.long)
        targets_tensor = torch.tensor(targets, dtype=torch.long)
        torch.save({"inputs": inputs_tensor, "targets": targets_tensor}, output_path)
        print(f"Saved preprocessed {split} data to {output_path}")

if __name__ == "__main__":
    preprocess_data(max_length=64)