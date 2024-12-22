import os
import re
import matplotlib.pyplot as plt

def parse_metrics(log_file):
    """
    metrics.log dosyasını okuyarak epoch başına metrikleri döndürür.
    """
    metrics = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_dice": [],
        "val_dice": []
    }

    with open(log_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            # Epoch numarasını bul
            epoch_match = re.search(r"Epoch (\d+):", line)
            if epoch_match:
                metrics["epoch"].append(int(epoch_match.group(1)))

            # Train Loss
            train_loss_match = re.search(r"'train_loss': ([0-9.]+)", line)
            if train_loss_match:
                metrics["train_loss"].append(float(train_loss_match.group(1)))

            # Validation Loss
            val_loss_match = re.search(r"'val_loss': ([0-9.]+)", line)
            if val_loss_match:
                metrics["val_loss"].append(float(val_loss_match.group(1)))

            # Train Dice
            train_dice_match = re.search(r"'train_dice': ([0-9.]+)", line)
            if train_dice_match:
                metrics["train_dice"].append(float(train_dice_match.group(1)))

            # Validation Dice
            val_dice_match = re.search(r"'val_dice': ([0-9.]+)", line)
            if val_dice_match:
                metrics["val_dice"].append(float(val_dice_match.group(1)))

    return metrics


def plot_metrics(metrics, output_dir):
    """
    Parse edilen metrikler için grafikler oluşturur ve kaydeder.
    """
    os.makedirs(output_dir, exist_ok=True)
    epochs = metrics["epoch"]

    # Loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics["train_loss"], label="Train Loss", marker='o')
    plt.plot(epochs, metrics["val_loss"], label="Validation Loss", marker='o')
    plt.title("Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "loss_plot.png"))
    plt.close()

    # Dice plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics["train_dice"], label="Train Dice", marker='o')
    plt.plot(epochs, metrics["val_dice"], label="Validation Dice", marker='o')
    plt.title("Dice Coefficient Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Dice Coefficient")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "dice_plot.png"))
    plt.close()


if __name__ == "__main__":
    # Yollar
    log_file = "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/checkpoints/metrics.log"
    output_dir = "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/visualizations"

    # Metrikleri parse et ve çiz
    metrics = parse_metrics(log_file)
    plot_metrics(metrics, output_dir)
    print(f"Metrics plots saved to {output_dir}")
