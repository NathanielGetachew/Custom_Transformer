# scripts/train_bpe_tokenizer.py
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
import os

def train_bpe_tokenizer(data_dir="data/ptbdataset", output_path="data/tokenizer", vocab_size=5000):
    # Initialize a BPE tokenizer
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel()

    # Define the trainer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["<pad>", "<sos>", "<eos>", "<unk>"]
    )

    # Train on the train split
    train_file = os.path.join(data_dir, "ptb.train.txt")
    tokenizer.train(files=[train_file], trainer=trainer)

    # Save the tokenizer
    os.makedirs(output_path, exist_ok=True)
    tokenizer.save(os.path.join(output_path, "tokenizer.json"))
    print(f"BPE tokenizer trained and saved to {output_path}/tokenizer.json")

if __name__ == "__main__":
    train_bpe_tokenizer()