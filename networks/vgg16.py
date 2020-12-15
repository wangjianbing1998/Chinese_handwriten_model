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

from keras.layers import Conv2D, Dense, Flatten,\
    Input, MaxPool2D
from keras.models import Model


class VGG16(BaseModel):

    def __init__(self, input_shape, nb_classes):
        super(VGG16, self).__init__(input_shape=input_shape, nb_classes=nb_classes)

    def build_model(self, input_shape, nb_classes=8):
        # 输入层
        img_input = Input(shape=input_shape, name='img_input')

        # 第1个卷积区块(block1)
        x = Conv2D(64, (3, 3), padding='same', activation='relu', name='block1_conv1')(img_input)
        x = Conv2D(64, (3, 3), padding='same', activation='relu', name='block1_conv2')(x)
        x = MaxPool2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # 第2个卷积区块(block2)
        x = Conv2D(128, (3, 3), padding='same', activation='relu', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), padding='same', activation='relu', name='block2_conv2')(x)
        x = MaxPool2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # 第3个卷积区块(block3)
        x = Conv2D(256, (3, 3), padding='same', activation='relu', name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), padding='same', activation='relu', name='block3_conv2')(x)
        x = Conv2D(256, (3, 3), padding='same', activation='relu', name='block3_conv3')(x)
        x = MaxPool2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # 第4个卷积区块(block4)
        x = Conv2D(512, (3, 3), padding='same', activation='relu', name='block4_conv1')(x)
        x = Conv2D(512, (3, 3), padding='same', activation='relu', name='block4_conv2')(x)
        x = Conv2D(512, (3, 3), padding='same', activation='relu', name='block4_conv3')(x)
        x = MaxPool2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # 第5个卷积区块(block5)
        x = Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_conv1')(x)
        x = Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_conv2')(x)
        x = Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_conv3')(x)
        x = MaxPool2D((2, 2), strides=(2, 2), name='block5_pool')(x)

        # 前馈全连接区块
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(1024, activation='relu', name='fc2')(x)
        x = Dense(nb_classes, activation='softmax', name='predictions')(x)

        # 产生模型
        model = Model(inputs=img_input, outputs=x, name='vgg16-funcapi')
        return model
