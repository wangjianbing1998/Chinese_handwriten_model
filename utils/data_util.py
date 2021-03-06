# encoding=utf-8
'''
@Author: jianbingxia 
@Time: 2020/12/11 14:48
@File: data_util.py
@Description: 



'''
import better_exceptions
import cv2
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.utils import Bunch

better_exceptions.hook()

import pandas as pd

pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 20)
pd.set_option('precision', 2)
import os
import numpy as np
import sklearn


class ResizeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if isinstance(X, np.ndarray):
            return np.resize(X, self.input_shape)


class Dataset():

    def __init__(self, config, shuffle=True, random_seed=100, feature_range=None):
        self.config = config
        self.data_dir = os.path.join(os.path.dirname(__file__), '../dataset/')

        df = pd.read_csv(os.path.join(self.data_dir, 'chinese_mnist.csv'))
        df = sklearn.utils.shuffle(df)
        images, labels = self.build_dataset(df, shuffle, random_seed)

        images_transformers = []
        if feature_range:
            images_transformers.append(MinMaxScaler(feature_range=feature_range))
        images, labels, self.categories_ = self.transform_data(
            images,
            images_transformers,
            labels,
            [OneHotEncoder()]
        )

        labels = labels.toarray()

        self.nb_classes = len(self.categories_)

        Xs_train, Xs_test, ys_train, ys_test = train_test_split(images, labels, test_size=.01)

        self.dataset = Bunch(
            Xs_train=Xs_train,
            Xs_test=Xs_test,
            ys_train=ys_train,
            ys_test=ys_test,
        )

    def build_dataset(self, df, shuffle=True, random_state=100):
        images = []
        labels = []
        for index in range(len(df)):

            if self.config.use_test_time and index == self.config.batch_size * 4:
                break

            path = os.path.join(self.data_dir, 'data/')
            image_path = path + 'input_' + str(df.iloc[index][0]) + "_" + str(df.iloc[index][1]) + "_" + str(
                df.iloc[index][2]) + ".jpg"

            if not os.path.exists(image_path):
                continue

            image = cv2.imread(image_path)
            if image.shape[-1] == 1:
                image_resized = cv2.resize(image.squeeze(), tuple(self.config.input_shape[:-1]))
                image_arr = cv2.cvtColor(image_resized, cv2.COLOR_GRAY2RGB)

            else:
                image_arr = np.resize(image, self.config.input_shape)

            images.append(image_arr)
            labels.append([df.iloc[index]['character']])

        images = np.array(images)
        labels = np.array(labels)

        if shuffle:
            images, labels = sklearn.utils.shuffle(images, labels, random_state=random_state)

        return images, labels

    def transform_data(self, images, image_transformers, labels, label_transformers):

        for transform in image_transformers:
            images_shape = images.shape
            x = images.reshape(images_shape[0], -1)
            x = transform.fit_transform(x)
            images = x.reshape(images_shape)

        for index, transform in enumerate(label_transformers):
            labels = transform.fit_transform(labels)
            if isinstance(transform, OneHotEncoder):
                # when transform the OneHotEncoder(), it must be last transformer for y
                if index != len(label_transformers) - 1:
                    raise ValueError('OneHotEncoder must be the last transformer for y')
                return images, labels, transform.categories_[0]
        return images, labels
