import os
import math
import random
import numpy as np
import pandas as pd
from imgaug import augmenters
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import Sequence


class DataGenerator(Sequence):

    def __init__(self, df, path_key, classes_key, batch_size=32, steps=None, shuffle=True):
        """ initialize image sequence generator 
        :param df: pandas dataframe, should contain columns of path_key and classes_key
        :param path_key: the column name of full path of image in df
        :param classes_key: the names of classes, in a list
        :param batch_size: batch size, int
        :param steps: number of batches to run in an epoch
        :param shuffle: flag to shuffle df rows at end of each epoch
        """
        self.df = df
        self.path_key = path_key
        self.classes_key = classes_key
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random.randint(0,1000)
        self.load_dataset()
        if steps is None:
            self.steps = math.ceil(self.x_path.shape[0] / (self.batch_size))
        else:
            self.steps = steps
        self.augmenter = augmenters.Sequential([ augmenters.Fliplr(0.5), ], random_order=True)  # image augmentation


    def __bool__(self):
        return True


    def __len__(self):
        """ denotes the number of batches per epoch """
        return self.steps


    def __getitem__(self, index):
        batch_x_path = self.x_path[index*self.batch_size : (index+1)*self.batch_size]
        batch_x = np.asarray([self.load_and_normalize(x_path) for x_path in batch_x_path])
        batch_x = self.transform_batch(batch_x)
        batch_y = self.y[index*self.batch_size : (index+1)*self.batch_size]
        return batch_x, batch_y


    def load_and_normalize(self, image_path, tmp_size=(256, 256), out_size=(224, 224)):
        # read image and resize to tmp size
        image = load_img(image_path, target_size=tmp_size)
        # print("PIL image size", image.size)
        # crop image center
        xoff = (tmp_size[0]-out_size[0])//2
        yoff = (tmp_size[1]-out_size[1])//2
        np_image = img_to_array(image)
        np_image = np_image[yoff:-yoff, xoff:-xoff]
        # normalize
        np_image = np_image / 255.
        # print("numpy array size", np_image.shape)
        # np_image = np.expand_dims(np_image, axis=0)
        return np_image


    def transform_batch(self, batch):
        # randomly horizontal flip images
        batch = self.augmenter.augment_images(batch)
        # normalize image, using the mean and standard deviation of imagenet dataset images
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])
        batch = (batch - imagenet_mean) / imagenet_std
        return batch


    def load_dataset(self):
        df = self.df.sample(frac=1, random_state=self.random_state)
        self.x_path, self.y = df[self.path_key].values(), df[self.classes_key].values()


    def on_epoch_end(self):
        if self.shuffle:
            self.random_state += 1
            self.load_dataset()
