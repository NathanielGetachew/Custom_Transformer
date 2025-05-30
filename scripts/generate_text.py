# scripts/generate_text.py
from src.data.dataset import load_and_preprocess
from src.models.transformer import TransformerDecoder
from src.training.generate import generate_text

def main():
    config = {
        "max_length": 128,
        "d_model": 256,
        "nhead": 8,
        "num_layers": 3,
        "dim_feedforward": 1024,
        "dropout": 0.1,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "final_model_path": "checkpoints/transformer_final.pth"
    }

    _, _, _, tokenizer = load_and_preprocess(max_length=config["max_length"])
    model = TransformerDecoder(
        vocab_size=tokenizer.vocab_size,
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        dim_feedforward=config["dim_feedforward"],
        max_len=config["max_length"],
        dropout=config["dropout"]
    ).to(config["device"])
    model.load_state_dict(torch.load(config["final_model_path"], map_location=config["device"]))

    prompts = ["the company said", "a new study", "in the market"]
    for prompt in prompts:
        generated = generate_text(model, tokenizer, config["device"], prompt)
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated}\n")

if __name__ == "__main__":
    main()