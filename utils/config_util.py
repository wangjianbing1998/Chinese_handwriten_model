# encoding=utf-8
'''
@Author: jianbingxia 
@Time: 2020/12/11 21:42
@File: config_util.py
@Description:

'''

import configparser
import logging

from sklearn.utils import Bunch

from utils.variable_util import TRAIN, TEST


class Configer():

    def __init__(self, config_path):
        self.config = configparser.ConfigParser()
        logging.info(f" Loading config file {config_path}...")
        self.config.read(config_path)

    def load_config(self, mode=TRAIN):
        if mode not in [TRAIN, TEST]:
            raise ValueError(f'Expected mode in ["train","test"], but got {mode}')
        return Bunch(**self.config[mode])
