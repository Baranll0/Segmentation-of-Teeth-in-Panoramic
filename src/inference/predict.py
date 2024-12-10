import torch
import matplotlib.pyplot as plt

def predict_and_visualize(model, dataset, idx, device, save_path=None):
    model.eval()
    image, mask = dataset[idx]
    image = image.unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        pred_mask = model(image).squeeze().cpu().numpy()

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Input Image")
    plt.imshow(image.squeeze().cpu().numpy(), cmap="gray")
    plt.subplot(1, 3, 2)
    plt.title("True Mask")
    plt.imshow(mask.squeeze().numpy(), cmap="gray")
    plt.subplot(1, 3, 3)
    plt.title("Predicted Mask")
    plt.imshow(pred_mask, cmap="gray")
    if save_path:
        plt.savefig(save_path)
    plt.show()
