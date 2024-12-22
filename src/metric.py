import os
import re
import matplotlib.pyplot as plt

def parse_metrics(log_file):
    """
    metrics.log dosyasını okuyarak epoch başına metrikleri döndürür.
    """
    metrics = {
        "epoch": [],
        "accuracy": [],
        "mean_dice": [],
        "mean_precision": [],
        "mean_recall": [],
        "mean_f1": [],
    }

    with open(log_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            # Epoch numarasını bul
            epoch_match = re.search(r"Epoch (\d+):", line)
            if epoch_match:
                metrics["epoch"].append(int(epoch_match.group(1)))

            # Accuracy
            accuracy_match = re.search(r"'accuracy': ([0-9.]+)", line)
            if accuracy_match:
                metrics["accuracy"].append(float(accuracy_match.group(1)))

            # Mean Dice
            mean_dice_match = re.search(r"'mean_dice': ([0-9.]+)", line)
            if mean_dice_match:
                metrics["mean_dice"].append(float(mean_dice_match.group(1)))

            # Mean Precision
            mean_precision_match = re.search(r"'mean_precision': ([0-9.]+)", line)
            if mean_precision_match:
                metrics["mean_precision"].append(float(mean_precision_match.group(1)))

            # Mean Recall
            mean_recall_match = re.search(r"'mean_recall': ([0-9.]+)", line)
            if mean_recall_match:
                metrics["mean_recall"].append(float(mean_recall_match.group(1)))

            # Mean F1
            mean_f1_match = re.search(r"'mean_f1': ([0-9.]+)", line)
            if mean_f1_match:
                metrics["mean_f1"].append(float(mean_f1_match.group(1)))

    return metrics

def plot_metrics(metrics, output_dir):
    """
    Parse edilen metrikler için grafikler oluşturur ve kaydeder.
    """
    os.makedirs(output_dir, exist_ok=True)
    epochs = metrics["epoch"]

    # Accuracy plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics["accuracy"], label="Accuracy", marker='o')
    plt.title("Accuracy Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "accuracy_plot.png"))
    plt.close()

    # Mean Dice plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics["mean_dice"], label="Mean Dice", marker='o')
    plt.title("Mean Dice Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Dice Coefficient")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "mean_dice_plot.png"))
    plt.close()

    # Mean Precision plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics["mean_precision"], label="Mean Precision", marker='o')
    plt.title("Mean Precision Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Precision")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "mean_precision_plot.png"))
    plt.close()

    # Mean Recall plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics["mean_recall"], label="Mean Recall", marker='o')
    plt.title("Mean Recall Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Recall")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "mean_recall_plot.png"))
    plt.close()

    # Mean F1 plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics["mean_f1"], label="Mean F1 Score", marker='o')
    plt.title("Mean F1 Score Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Mean F1 Score")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "mean_f1_plot.png"))
    plt.close()

if __name__ == "__main__":
    # Yollar
    log_file = "/nested-unet/metrics.log"
    output_dir = "/media/baran/Disk1/Segmentation-of-Teeth-in-Panoramic/visualizations"

    # Metrikleri parse et ve çiz
    metrics = parse_metrics(log_file)
    plot_metrics(metrics, output_dir)
    print(f"Metrics plots saved to {output_dir}")
