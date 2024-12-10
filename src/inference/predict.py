import numpy as np
import matplotlib.pyplot as plt

def predict_and_visualize(model, X_test, y_test, idx):
    pred_mask = model.predict(np.expand_dims(X_test[idx], axis=0))[0]
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title('Input Image')
    plt.imshow(X_test[idx].squeeze(), cmap='gray')
    plt.subplot(1, 3, 2)
    plt.title('True Mask')
    plt.imshow(y_test[idx].squeeze(), cmap='gray')
    plt.subplot(1, 3, 3)
    plt.title('Predicted Mask')
    plt.imshow(pred_mask.squeeze(), cmap='gray')
    plt.show()
