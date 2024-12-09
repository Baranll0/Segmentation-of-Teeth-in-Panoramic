import torch
import numpy as np

def postprocess_segmentation(predictions, threshold=0.5):
    predictions = torch.sigmoid(predictions)
    predictions = (predictions > threshold).float()
    return predictions.cpu().numpy()