import os
import json
from itertools import chain
from keras import optimizers
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint


csv_file = "../../CHESTXRAY/Data_Entry_2017.csv"
csv_file_encoded = "Data_Entry_2017_Encoded.csv"
image_dir = "../../images"
image_col = "Image Index"  # image name column in csv
classes = ['Atelectasis','Cardiomegaly','Consolidation','Edema','Effusion',
           'Emphysema','Fibrosis','Hernia','Infiltration','Mass','Nodule',
           'Pleural_Thickening','Pneumonia','Pneumothorax']
nb_classes = len(classes)

weights_pretrained = "imagenet"
image_shape = (299, 299)
input_shape = (299, 299, 3)
slices = [0.8, 0.1, 0.1]  # train/valid/test
epochs = 50
batch_size = 32


def build_model():
    base_model = InceptionV3(weights=weights_pretrained, include_top=False)

    nb_gpus = len(os.getenv("CUDA_VISIBLE_DEVICES", "1").split(","))
    input_tensor = Input(input_shape)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(2048, activation="relu")(x)
    predictions = Dense(nb_classes, outputs=predictions)

    model = Model(inputs=input_tensor, outputs=predictions)
    if nb_gpus > 1:
            model = multi_gpu_model(model, gpus=nb_gpus)

    return model


def get_generator(dataframe, horizontal_flip=False, shuffle=True):
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input, horizontal_flip=horizontal_flip)
    generator = datagen.flow_from_dataframe(dataframe=dataframe, 
                                            directory=image_dir, 
                                            x_col=image_col, 
                                            y_col=classes, 
                                            has_ext=True, 
                                            target_size=image_shape,  # (height, weight)
                                            batch_size=batch_size, 
                                            shuffle=shuffle,          # shuffle images
                                            class_mode="categorical", # class mode
                                            save_to_dir=None)         # save augmented images to local
    return generator


def save_history(history):
    with open("history.json", "wb") as f:
        json.dump(history.history, f)


def train(dataframe, model):
    nb_records, _ = dataframe.shape
    train_generator = get_generator(dataframe=dataframe.iloc[:int(nb_records*slices[0])], 
                                    horizontal_flip=True, 
                                    shuffle=True)
    valid_generator = get_generator(dataframe=dataframe.iloc[int(nb_records*slices[0]) : int(nb_records*(slices[0]+slices[1]))], 
                                    horizontal_flip=True, 
                                    shuffle=True)

    checkpoint = ModelCheckpoint("weights_{epoch:03d}_{val_acc:.4f}.hdf5", 
                                 monitor="val_acc", 
                                 verbose=1, 
                                 save_best_only=False, 
                                 mode="auto", 
                                 save_weights_only=True)
    adam = optimizers.Adam(lr=0.001)
    model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"])
    history = model.fit_generator(train_generator, epochs=epochs, validation_data=valid_generator, callbacks=[checkpoint])

    save_history(history)


def prepare_dataframe():
    df = pd.read_csv(csv_file)
    df = df.sample(frac=1)  # shuffle rows in dataframe
    all_labels = np.unique(list(chain(*df["Finding Labels"].map(lambda x: x.split('|')).tolist())))
    for label in all_labels:
        df[label] = df["Finding Labels"].map(lambda x: 1 if label in x else 0)  # one-hot encoding
    df.to_csv(csv_file_encoded)
    return df


if __name__ == "__main__":
    df = prepare_dataframe()
    model = build_model()
    train(dataframe=df, model=model)
