# encoding=utf-8
'''
@Author: jianbingxia 
@Time: 2020/12/11 14:48
@File: data_util.py
@Description: 



'''
import better_exceptions
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import Bunch

better_exceptions.hook()

import pandas as pd

pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 20)
pd.set_option('precision', 2)
import os
import cv2
import numpy as np
import sklearn


class ResizeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, target_size):
        self.target_size = target_size

    def fit(self):
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            X.resize(self.target_size)


class Dataset():

    def __init__(self, config, shuffle=True, random_state=100):
        self.config = config
        self.data_dir = os.path.join(os.path.dirname(__file__), '../dataset/')

        df = pd.read_csv(os.path.join(self.data_dir, 'chinese_mnist.csv'))
        images, labels = self.build_dataset(df, shuffle, random_state)

        images, labels, self.categories_ = self.transform_data(images,
                                                               [ResizeTransformer(target_size=config.target_size)],
                                                               labels, [OneHotEncoder()])

        Xs_train, Xs_test, ys_train, ys_test = train_test_split(images, labels)

        self.dataset = Bunch(
            Xs_train=Xs_train,
            Xs_test=Xs_test,
            ys_train=ys_train,
            ys_test=ys_test,
        )
        self.nb_classes = len(self.categories_)

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

    def transform_data(self, images, image_transformers, labels, label_transformers):

        for transform in image_transformers:
            images = transform.fit_transform(images)

        for index, transform in enumerate(label_transformers):
            labels = transform.fit_transform(labels)
            if isinstance(transform, OneHotEncoder):
                # when transform the OneHotEncoder(), it must be last transformer for y
                if index != len(self.y_transforms) - 1:
                    raise ValueError('OneHotEncoder must be the last transformer for y')
                return images, labels, transform.categories_[0]
        return images, labels
