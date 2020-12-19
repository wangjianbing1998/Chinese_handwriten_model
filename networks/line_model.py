# encoding=utf-8
'''
@Author: jianbingxia 
@Time: 2020/12/11 22:19
@File: simple_model.py
@Description:
'''
import better_exceptions
from keras import Sequential
from keras.initializers import Constant

from networks.base_model import BaseModel

better_exceptions.hook()

from keras.layers import Conv2D, Dense, MaxPool2D, Flatten, PReLU, Dropout


class LineModel(BaseModel):

    def __init__(self, input_shape, nb_classes):
        super(LineModel, self).__init__(input_shape=input_shape, nb_classes=nb_classes)

    def build_model(self, input_shape, nb_classes=8):
        model = Sequential()
        model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=input_shape, padding='same'))
        model.add(PReLU(alpha_initializer=Constant(value=0.25)))
        model.add(MaxPool2D(2))

        model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same'))
        model.add(PReLU(alpha_initializer=Constant(value=0.25)))
        model.add(MaxPool2D(2))

        model.add(Conv2D(filters=160, kernel_size=(3, 3), padding='same'))
        model.add(PReLU(alpha_initializer=Constant(value=0.25)))
        model.add(MaxPool2D(2))

        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same'))
        model.add(PReLU(alpha_initializer=Constant(value=0.25)))
        model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same'))
        model.add(PReLU(alpha_initializer=Constant(value=0.25)))
        model.add(MaxPool2D(2))

        model.add(Conv2D(filters=384, kernel_size=(3, 3), padding='same'))
        model.add(PReLU(alpha_initializer=Constant(value=0.25)))
        model.add(Conv2D(filters=384, kernel_size=(3, 3), padding='same'))
        model.add(PReLU(alpha_initializer=Constant(value=0.25)))
        model.add(MaxPool2D(2))

        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(PReLU(alpha_initializer=Constant(value=0.25)))
        model.add(Dropout(0.5))
        model.add(Dense(15, activation='softmax'))
        model.add(PReLU(alpha_initializer=Constant(value=0.25)))

        return model
