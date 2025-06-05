# scripts/evaluate_model.py
from src.data.dataset import load_and_preprocess
from src.models.transformer import TransformerDecoder
from src.training.evaluate import evaluate_model
from torch.utils.data import DataLoader
def main():
    config = {
        "batch_size": 4,
        "max_length": 64,  # Match train_model.py
        "d_model": 128,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 512,
        "dropout": 0.1,
        "device": "cpu",
        "final_model_path": "checkpoints/transformer_final.pth"
    }

    _, _, test_data, tokenizer = load_and_preprocess(max_length=config["max_length"])
    test_loader = DataLoader(test_data, batch_size=config["batch_size"], shuffle=False, num_workers=2, pin_memory=config["device"].startswith("cuda"))

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

    test_loss, test_perplexity = evaluate_model(model, test_loader, tokenizer, config["device"])
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Perplexity: {test_perplexity:.2f}")

if __name__ == "__main__":
    main()