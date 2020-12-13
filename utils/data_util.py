# encoding=utf-8
'''
@Author: jianbingxia 
@Time: 2020/12/11 14:48
@File: data_util.py
@Description: 



'''
import better_exceptions
from sklearn.preprocessing import OneHotEncoder

better_exceptions.hook()

import pandas as pd

pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 20)
pd.set_option('precision', 2)
import os
import cv2
import numpy as np
import sklearn


class Dataset():

    def __init__(self, config, shuffle=True, random_state=100):
        self.config = config
        self.data_dir = os.path.join(os.path.dirname(__file__), '../dataset/')

        df = pd.read_csv(os.path.join(self.data_dir, 'chinese_mnist.csv'))
        images, labels = self.build_dataset(df, shuffle, random_state)

        self.x_transforms = []
        self.y_transforms = [OneHotEncoder()]

        images, labels, self.categories_ = self.transform_data(images, labels)

    def build_dataset(self, df, shuffle=True, random_state=100):
        images = []
        labels = []
        for index in range(len(df)):
            path = os.path.join(self.data_dir, 'data/')
            image_path = path + 'input_' + str(df.iloc[index][0]) + "_" + str(df.iloc[index][1]) + "_" + str(
                df.iloc[index][2]) + ".jpg"
            image_arr = cv2.imread(image_path)
            images.append(image_arr)
            labels.append([df.iloc[index]['character']])

        images = np.array(images)
        labels = np.array(labels)

        if shuffle:
            images, labels = sklearn.utils.shuffle(images, labels, random_state=random_state)

        return images, labels

    def transform_data(self, images, labels):

        for transform in self.x_transforms:
            images = transform.fit_transform(images)

        for index, transform in enumerate(self.y_transforms):
            labels = transform.fit_transform(labels)
            if isinstance(transform, OneHotEncoder):
                # when transform the OneHotEncoder(), it must be last transformer for y
                if index != len(self.y_transforms) - 1:
                    raise ValueError('OneHotEncoder must be the last transformer for y')
                return images, labels, transform.categories_
        return images, labels
