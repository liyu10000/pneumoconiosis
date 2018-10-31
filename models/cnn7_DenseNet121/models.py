import os
from keras.applications.densenet import DenseNet121
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.layers import Input
from keras.layers.core import Dense
from keras.models import Model
from keras.utils import multi_gpu_model


class ModelFactory:
    def __init__(self, nb_classes, input_shape):
        self.nb_classes = nb_classes
        self.input_shape = input_shape
        
    def densenet121(self):
        nb_gpus = len(os.getenv("CUDA_VISIBLE_DEVICES", "1").split(","))
        input_tensor = Input(shape=self.input_shape)

        base_model = DenseNet121(include_top=False,
                                 input_tensor=input_tensor, 
                                 input_shape=self.input_shape, 
                                 weights='imagenet', 
                                 pooling="avg")
        x = base_model.output
        predictions = Dense(self.nb_classes, activation="sigmoid", name="predictions")(x)
        model = Model(inputs=input_tensor, outputs=predictions)
        if nb_gpus > 1:
            model = multi_gpu_model(model, gpus=nb_gpus)
        # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        return model
    
    def inceptionv3(self):
        nb_gpus = len(os.getenv("CUDA_VISIBLE_DEVICES", "1").split(","))
        input_tensor = Input(shape=self.input_shape)
        
        base_model = InceptionV3(weights=weights_pretrained, include_top=False)

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(2048, activation="relu")(x)
        predictions = Dense(nb_classes, activation="softmax", name="predictions")(x)

        model = Model(inputs=input_tensor, outputs=predictions)
        if nb_gpus > 1:
                model = multi_gpu_model(model, gpus=nb_gpus)

        return model



if __name__ == "__main__":
    model = ModelFactory(2, (224,224,3)).inceptionv3()
    model.summary()