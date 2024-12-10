import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from src.models.unet import UNET
from src.evaluation.metrics import dice_loss, dice_coefficient
import tensorflow as tf

def train_model(X_train, y_train, X_val, y_val, batch_size=8, epochs=100, learning_rate=1e-4):
    model = UNET(input_shape=(512, 512, 1), last_activation='sigmoid')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=dice_loss,
                  metrics=[dice_coefficient, 'accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=[early_stopping, reduce_lr])
    return model, history
