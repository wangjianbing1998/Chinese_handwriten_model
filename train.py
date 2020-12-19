# encoding=utf-8
'''
@Author: jianbingxia 
@Time: 2020/12/13 17:40
@File: train.py
@Description: 

'''
import os
import warnings

from keras.preprocessing.image import ImageDataGenerator

from utils.model_util import get_model

warnings.filterwarnings('ignore')
from collections import ChainMap

import better_exceptions
from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from sklearn.utils import Bunch

from utils.config_util import Configer
from utils.data_util import Dataset
from utils.variable_util import TRAIN, TEST_TIME

better_exceptions.hook()

configer = Configer("configs/config.ini")
config = configer.load_config(TRAIN)
import argparse

parse = argparse.ArgumentParser('Writen Chinese Number Recognition')
parse.add_argument('-t', '--use_test_time', action='store_true',
                   help='whether to use test, switch test-time training')
parse.add_argument('-m', '--model', default='vgg16',
                   help='which model to train')
args = parse.parse_args()

config = Bunch(**ChainMap(vars(args), config))

if config.use_test_time:
    use_test_time_config = configer.load_config(TEST_TIME)
    config = Bunch(**ChainMap(use_test_time_config, config))

Configer.print_config(config)

dataset = Dataset(
    config=config,
    shuffle=True,
    feature_range=(0, 1),
    random_seed=config.random_seed,
)

config['nb_classes'] = dataset.nb_classes

model = get_model(config)

data = dataset.dataset

model_callbacks = [
    ModelCheckpoint(
        os.path.join(config.model_path, config.model),
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        mode='auto',
        period=1,
    ),
    # EarlyStopping(monitor='val_loss', min_delta=0),
    TensorBoard(log_dir='./logs', batch_size=config.batch_size),
    LearningRateScheduler(lambda x:1e-3 * 0.99 ** (x + config.epochs))
]

#
# estimator = KerasClassifier(build_fn=lambda :model, nb_epoch=config.epochs, batch_size=config.batch_size)
#
# np.random.seed(config.random_seed)
# kfold = KFold(n_splits=config.n_splits, shuffle=config.shuffle, random_state=config.random_seed)
# results = cross_val_score(estimator, data.Xs_train, data.ys_train, cv=kfold)


datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

model.fit_generator(datagen.flow(data.Xs_train, data.ys_train,
                                 batch_size=config.batch_size),

                    steps_per_epoch=len(data.Xs_train),
                    epochs=config.epochs,
                    validation_data=(data.Xs_test, data.ys_test),
                    callbacks=model_callbacks
                    )

# model.fit(
#     data.Xs_train,
#     data.ys_train,
#     validation_split=.1,
#     batch_size=config.batch_size,
#     epochs=config.epochs,
#     verbose=1,
#     callbacks=model_callbacks,
# )
