from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.layers import (Conv2D, Conv2DTranspose, Dense, Flatten,
                                     Input, MaxPooling2D, concatenate, BatchNormalization, PReLU)
from tensorflow.keras.models import Model

def regressor_mobilenet():

    backbone = MobileNetV3Small(minimalistic = True, include_top = False)

    input_layer = Input(shape = (64, 64, 3))
    x = backbone(input_layer)
    x = Flatten()(x)

    x = Dense(256)(x)
    x = PReLU()(x)

    x = Dense(64)(x)
    x = PReLU()(x)

    output_layer = Dense(8, activation = 'sigmoid')(x)

    model = Model(input_layer, output_layer)
    return model

def unet_downsample(x, num_filters, batch_norm = False):

    x = Conv2D(num_filters, (3, 3), padding = 'same',
               activation = 'relu')(x)

    if batch_norm: x = BatchNormalization()(x)
    
    x = Conv2D(num_filters, (3, 3), padding = 'same',
               activation = 'relu')(x)

    if batch_norm: x = BatchNormalization()(x)

    x_down = MaxPooling2D()(x)

    return x_down, x

def unet_upsample(x, x_down, num_filters, batch_norm = False):

    x = Conv2DTranspose(num_filters, (3, 3), strides = (2, 2), padding = 'same',
                        activation = 'relu')(x)

    if batch_norm: x = BatchNormalization()(x)
    
    x = concatenate([x, x_down])

    x = Conv2D(num_filters, (3, 3), padding = 'same',
               activation = 'relu')(x)

    if batch_norm: x = BatchNormalization()(x)

    x = Conv2D(num_filters, (3, 3), padding = 'same',
               activation = 'relu')(x)

    if batch_norm: x = BatchNormalization()(x)

    return x

def regressor_hmaps_unet(s_filters = 8, levels = 4, batch_norm = False):

    input_layer = Input(shape = (64, 64, 3))

    # Mimic U-net architecture

    x = []

    down = input_layer
    for i in range(levels):
        down, x_i = unet_downsample(down, s_filters * 2**i, batch_norm)
        x.append(x_i)

    x_l = Conv2D(s_filters * 2**levels, (3, 3), padding = 'same',
                 activation = 'relu')(down)

    if batch_norm: x_l = BatchNormalization()(x_l)

    x_l = Conv2D(s_filters * 2**levels, (3, 3), padding = 'same',
                 activation = 'relu')(x_l)

    if batch_norm: x_l = BatchNormalization()(x_l)

    up = x_l
    for i in reversed(range(levels)):
        x_i = x.pop()
        up = unet_upsample(up, x_i, s_filters * 2**i, batch_norm)

    output_layer = Conv2D(1, (3, 3), padding = 'same',
    activation = 'sigmoid')(up)

    model = Model(input_layer, output_layer)
    return model

def simple_decoder():

    input_layer = Input(shape = (32, 32, 1))

    x = Conv2D(8, (5, 5), padding = 'valid')(input_layer)
    x = PReLU()(x)
    x = MaxPooling2D()(x)

    x = Conv2D(4, (3, 3), padding = 'valid')(x)
    x = PReLU()(x)
    x = MaxPooling2D()(x)

    output_layer = Conv2D(1, (1, 1), padding = 'valid', 
    activation = 'sigmoid')(x)

    model = Model(input_layer, output_layer)
    return model