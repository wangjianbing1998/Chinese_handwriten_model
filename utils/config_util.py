# encoding=utf-8
'''
@Author: jianbingxia 
@Time: 2020/12/11 21:42
@File: config_util.py
@Description:

'''

import configparser
import logging


def get_config(config_path="../configs/data.ini"):
    config = configparser.ConfigParser()
    logging.info(f" Loading config file {config_path}...")
    config.read(config_path)
