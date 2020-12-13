# encoding=utf-8
'''
@Author: jianbingxia 
@Time: 2020/12/13 17:40
@File: train.py
@Description: 



'''
import better_exceptions
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau

from networks.resNet34 import ResNet34
from utils.config_util import Configer
from utils.data_util import Dataset
from utils.variable_util import TRAIN

better_exceptions.hook()

configer = Configer("configs/data.ini")
config = configer.load_config(TRAIN)

print(config)
dataset = Dataset(
    config=config,
    shuffle=True,
    random_state=100,
)
model = ResNet34(
    target_size=224,
    nb_classes=dataset.nb_classes,
).model

data = dataset.dataset

model_callbacks = [
    ModelCheckpoint(
        config.model_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        mode='auto',
        period=1,
    ),
    EarlyStopping(monitor='val_loss', min_delta=0),
    TensorBoard(log_dir='./logs', batch_size=config.batch_size),
    ReduceLROnPlateau(monitor='val_loss',
                      factor=.1,
                      pations=10,
                      mode='auto',
                      )
]

model.fit(
    data.Xs_train,
    data.ys_train,
    validation_split=.15,
    batch_size=config.batch_size,
    epochs=config.epochs,
    verbose=1,
    callbacks=model_callbacks,
)
