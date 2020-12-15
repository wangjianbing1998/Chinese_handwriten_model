# encoding=utf-8
'''
@Author: jianbingxia 
@Time: 2020/12/11 22:19
@File: simple_model.py
@Description:
'''
import better_exceptions

from networks.base_model import BaseModel

better_exceptions.hook()

from keras.layers import Conv2D, Dense, Input, MaxPool2D, Flatten
from keras.models import Model


class SimpleModel(BaseModel):

    def __init__(self, input_shape, nb_classes):
        super(SimpleModel, self).__init__(input_shape=input_shape, nb_classes=nb_classes)

    def build_model(self, input_shape, nb_classes=8):
        img_input = Input(shape=input_shape, name='img_input')  # 64

        x = Conv2D(6, (3, 3), padding='same', activation='relu', name='block1_conv1')(img_input)  # 64
        x = MaxPool2D((2, 2), strides=(2, 2), name='block1_pool')(x)  # 32
        x = Conv2D(16, (3, 3), padding='same', activation='relu', name='block2_conv1')(x)  # 32
        x = MaxPool2D((2, 2), strides=(2, 2), name='block2_pool')(x)  # 16

        x = Flatten(name='flatten')(x)
        x = Dense(128, activation='relu', name='fc1')(x)
        x = Dense(64, activation='relu', name='fc2')(x)
        x = Dense(nb_classes, activation='softmax', name='predictions')(x)

        model = Model(inputs=img_input, outputs=x, name='simple-model')
        return model
