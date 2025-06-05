import matplotlib.pyplot as plt

def plot_losses(steps, train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(steps, train_losses, label="Train Loss")
    plt.plot(steps, val_losses, label="Validation Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_plot.png")
    plt.close()