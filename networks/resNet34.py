# encoding=utf-8
'''
@Author: jianbingxia 
@Time: 2020/12/11 22:19
@File: resNet34.py
@Description:
'''
import better_exceptions
from keras_contrib import applications

from networks.base_model import BaseModel

better_exceptions.hook()

from keras.layers import Conv2D, BatchNormalization, add


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
        return applications.ResNet34(input_shape, nb_classes)
