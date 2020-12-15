# encoding=utf-8
'''
@Author: jianbingxia 
@Time: 2020/12/13 17:03
@File: base_model.py
@Description: 



'''
import better_exceptions

better_exceptions.hook()


class BaseModel():
    def __init__(self, input_shape=[224, 224, 3], nb_classes=8):
        self.model = self.build_model(input_shape=input_shape, nb_classes=nb_classes)
        # Print the detail of the model
        self.model.summary()
        # compile the model
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['acc'])
        # plot_model(self.model, to_file='networks/model_ResNet-34.png')

    def build_model(self, input_shape, nb_classes):
        pass
