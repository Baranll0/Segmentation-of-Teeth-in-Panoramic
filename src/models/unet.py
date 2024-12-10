from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, concatenate, Conv2DTranspose, \
    Dropout
from tensorflow.keras.models import Model


def UNET(input_shape=(512, 512, 1), last_activation='sigmoid'):
    inputs = Input(shape=input_shape)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    d1 = Dropout(0.1)(conv1)
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(d1)
    b1 = BatchNormalization()(conv2)
    pool1 = MaxPooling2D(pool_size=(2, 2))(b1)

    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    d2 = Dropout(0.2)(conv3)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(d2)
    b2 = BatchNormalization()(conv4)
    pool2 = MaxPooling2D(pool_size=(2, 2))(b2)

    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    d3 = Dropout(0.3)(conv5)
    conv6 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(d3)
    b3 = BatchNormalization()(conv6)
    pool3 = MaxPooling2D(pool_size=(2, 2))(b3)

    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    d4 = Dropout(0.4)(conv7)
    conv8 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(d4)
    b4 = BatchNormalization()(conv8)

    up1 = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(b4)
    merge1 = concatenate([up1, b3])
    conv9 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(merge1)
    conv10 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    up2 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(conv10)
    merge2 = concatenate([up2, b2])
    conv11 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(merge2)
    conv12 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv11)

    up3 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(conv12)
    merge3 = concatenate([up3, b1])
    conv13 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(merge3)
    conv14 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv13)

    outputs = Conv2D(1, (1, 1), activation=last_activation)(conv14)
    model = Model(inputs=inputs, outputs=outputs)
    return model