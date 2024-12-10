import numpy as np
import cv2

def postprocess_predictions(predictions, min_size=500):
    processed = []
    for pred in predictions:
        mask = (pred > 0.5).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = remove_small_objects(mask, min_size)
        processed.append(mask)
    return np.array(processed)

def remove_small_objects(mask, min_size):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype('uint8'))
    cleaned_mask = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            cleaned_mask[labels == i] = 1
    return cleaned_mask