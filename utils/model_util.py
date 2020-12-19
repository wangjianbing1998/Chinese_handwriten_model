# encoding=utf-8
'''
@Author: jianbingxia 
@Time: 2020/12/15 11:18
@File: model_util.py
@Description: 


'''

from networks.line_model import LineModel
from networks.resNet34 import ResNet34
from networks.simple_model import SimpleModel
from networks.vgg16 import VGG16
from networks.vgg19 import VGG19


def get_model(config):
    if config.model == 'vgg16':
        return VGG16(
            input_shape=config.input_shape,
            nb_classes=config.nb_classes,
        ).model
    elif config.model == 'vgg19':
        return VGG19(
            input_shape=config.input_shape,
            nb_classes=config.nb_classes,
        ).model

    elif config.model == 'resnet34':
        return ResNet34(
            input_shape=config.input_shape,
            nb_classes=config.nb_classes,
        ).model

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
