# encoding=utf-8
'''
@Author: jianbingxia 
@Time: 2020/12/11 22:19
@File: vgg19.py
@Description:
'''
import better_exceptions

from networks.base_model import BaseModel

better_exceptions.hook()

from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
import keras


class VGG19(BaseModel):

    def __init__(self, input_shape, nb_classes):
        super(VGG19, self).__init__(input_shape=input_shape, nb_classes=nb_classes)

    def build_model(self, input_shape=(48, 48, 3), nb_classes=8):
        base_model = keras.applications.vgg19.VGG19(include_top=False, input_shape=input_shape)
        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(.6)(x)
        x = Dense(512, activation='relu')(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(.5)(x)
        predictions = Dense(nb_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)

        return model
