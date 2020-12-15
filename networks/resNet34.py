# encoding=utf-8
'''
@Author: jianbingxia 
@Time: 2020/12/11 22:19
@File: resNet34.py
@Description:
'''
import better_exceptions

from networks.base_model import BaseModel

better_exceptions.hook()

from keras.layers import Conv2D, BatchNormalization, Dense, Flatten,\
    MaxPooling2D, AveragePooling2D, ZeroPadding2D, Input, add
from keras.models import Model


class ResNet34(BaseModel):

    def __init__(self, input_shape, nb_classes):
        super(ResNet34, self).__init__(input_shape=input_shape, nb_classes=nb_classes)

    def _conv_block(self, inputs,
                    neuron_num,
                    kernel_size,
                    use_bias,
                    padding='same',
                    strides=(1, 1),
                    with_conv_short_cut=False):
        conv1 = Conv2D(
            neuron_num,
            kernel_size=kernel_size,
            activation='relu',
            strides=strides,
            use_bias=use_bias,
            padding=padding
        )(inputs)
        conv1 = BatchNormalization(axis=1)(conv1)

        conv2 = Conv2D(
            neuron_num,
            kernel_size=kernel_size,
            activation='relu',
            use_bias=use_bias,
            padding=padding)(conv1)
        conv2 = BatchNormalization(axis=1)(conv2)

        if with_conv_short_cut:
            inputs = Conv2D(
                neuron_num,
                kernel_size=kernel_size,
                strides=strides,
                use_bias=use_bias,
                padding=padding
            )(inputs)
            return add([inputs, conv2])

        else:
            return add([inputs, conv2])

    def build_model(self, input_shape, nb_classes=8):
        inputs = Input(shape=input_shape)
        x = ZeroPadding2D((3, 3))(inputs)
        # Define the converlutional block 1
        x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='valid')(x)
        x = BatchNormalization(axis=1)(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
        # Define the converlutional block 2
        x = self._conv_block(x, neuron_num=64, kernel_size=(3, 3), use_bias=True)
        x = self._conv_block(x, neuron_num=64, kernel_size=(3, 3), use_bias=True)
        x = self._conv_block(x, neuron_num=64, kernel_size=(3, 3), use_bias=True)
        # Define the converlutional block 3
        x = self._conv_block(x, neuron_num=128, kernel_size=(3, 3), use_bias=True, strides=(2, 2),
                             with_conv_short_cut=True)
        x = self._conv_block(x, neuron_num=128, kernel_size=(3, 3), use_bias=True)
        x = self._conv_block(x, neuron_num=128, kernel_size=(3, 3), use_bias=True)
        # Define the converlutional block 4
        x = self._conv_block(x, neuron_num=256, kernel_size=(3, 3), use_bias=True, strides=(2, 2),
                             with_conv_short_cut=True)
        x = self._conv_block(x, neuron_num=256, kernel_size=(3, 3), use_bias=True)
        x = self._conv_block(x, neuron_num=256, kernel_size=(3, 3), use_bias=True)
        x = self._conv_block(x, neuron_num=256, kernel_size=(3, 3), use_bias=True)
        x = self._conv_block(x, neuron_num=256, kernel_size=(3, 3), use_bias=True)
        x = self._conv_block(x, neuron_num=256, kernel_size=(3, 3), use_bias=True)
        # Define the converltional block 5
        x = self._conv_block(x, neuron_num=512, kernel_size=(3, 3), use_bias=True, strides=(2, 2),
                             with_conv_short_cut=True)
        x = self._conv_block(x, neuron_num=512, kernel_size=(3, 3), use_bias=True)
        x = self._conv_block(x, neuron_num=512, kernel_size=(3, 3), use_bias=True)
        x = AveragePooling2D(pool_size=(7, 7))(x)
        x = Flatten()(x)
        x = Dense(nb_classes, activation='softmax')(x)
        model = Model(inputs=inputs, outputs=x)
        return model
