import os
from keras.applications.densenet import DenseNet121
from keras.layers import Input
from keras.layers.core import Dense
from keras.models import Model
from keras.utils import multi_gpu_model


class ModelFactory:
    def __init__(self, nb_classes, input_shape, nb_gpus=1):
        self.nb_classes = nb_classes
        self.input_shape = input_shape
        self.nb_gpus = nb_gpus
        
    def densenet121(self):
        input_tensor = Input(shape=self.input_shape)

        base_model = DenseNet121(include_top=False,
                                 input_tensor=input_tensor, 
                                 input_shape=self.input_shape, 
                                 weights='imagenet',  # use pretrained weights
                                 pooling="avg")
        x = base_model.output
        predictions = Dense(self.nb_classes, activation="sigmoid", name="predictions")(x)
        model = Model(inputs=input_tensor, outputs=predictions)
        if self.nb_gpus > 1:
            model = multi_gpu_model(model, gpus=self.nb_gpus)
        # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        return model



if __name__ == "__main__":
    model = ModelFactory(2, (224,224,3)).densenet121()
    model.summary()