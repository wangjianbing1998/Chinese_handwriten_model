# encoding=utf-8
'''
@Author: jianbingxia 
@Time: 2020/12/15 11:18
@File: model_util.py
@Description: 



'''
# from networks.resNet34 import ResNet34
# from networks.vgg16 import SimpleModel
#
#
# def get_model(config):
#     if config.model == 'vgg16':
#         return VGG16(
#             input_shape=config.input_shape,
#             nb_classes=config.nb_classes,
#         ).model
#     elif config.model == 'resnet34':
#         return ResNet34(
#             input_shape=config.input_shape,
#             nb_classes=config.nb_classes,
#         ).model
from keras import Model
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, GlobalAveragePooling2D
from keras.metrics import top_k_categorical_accuracy

from networks.line_model import LineModel
from networks.simple_model import SimpleModel


def get_model(config):
    if config.model == 'vgg16':
        base_model = VGG16(include_top=False, input_shape=config.input_shape)
        # 添加全局平均池化层
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(config.nb_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.summary()
        # compile the model
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['acc', top_k_categorical_accuracy])

        return model
    elif config.model == 'resnet34':
        pass

    elif config.model == 'simple':
        return SimpleModel(
            input_shape=config.input_shape,
            nb_classes=config.nb_classes,
        ).model

    elif config.model == 'line':
        return LineModel(
            input_shape=config.input_shape,
            nb_classes=config.nb_classes,
        ).model
