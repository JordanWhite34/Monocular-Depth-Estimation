import os
import sys

import tensorflow as tf
from tensorflow.keras import layers

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

tf.random.set_seed(123)

# downloading the dataset
annotation_folder = "/dataset/"
if not os.path.exists(os.path.abspath(".") + annotation_folder):
    annotation_zip = tf.keras.utils.get_file(
        "val.tar.gz",
        cache_subdir=os.path.abspath("."),
        origin="http://diode-dataset.s3.amazonaws.com/val.tar.gz",
        extract=True
    )

# prepping dataset
path = "val/indoors"

file_list = []

for root, _, files in os.walk(path):
    for file in files:
        file_list.append(os.path.join(root, file))

## sorting the file list
file_list.sort()
data = {
    "image": [x for x in file_list if x.endswith(".png")],
    "depth": [x for x in file_list if x.endswith("_depth.npy")],
    "mask": [x for x in file_list if x.endswith("_depth_mask.npy")]
}

## making dataframe for the sorted dataset
df = pd.DataFrame(data)
df = df.sample(frac=1, random_state=42)


# hyperparameter setup
HEIGHT = 256
WIDTH = 256
LR = 0.0002
EPOCHS = 30
BATCH_SIZE = 32


# building data pipeline
## - input: dataframe containing path for RGB images, depth, and depth mask files
## - reads and resizes RGB images
## - reads depth, depth mask files and processes them to generate depth map image and resizes it
## - returns: RGB images and depth map images for a batch

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data, batch_size=6, dim=(768, 1024), n_channels=3, shuffle=True):
        # initialization
        self.data = data
        self.indices = self.data.index.tolist()
        self.dim = dim
        self.n_channels = n_channels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.min_depth = 0.1
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        if (index + 1) * self.batch_size > len(self.indices):
            self.batch_size = len(self.indices) - index * self.batch_size
        # generate 1 data batch
        # generate batch indices
        index = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        # find list of IDs
        batch = [self.indices[k] for k in index]
        x, y = self.data_generation(batch)

        return x, y

    def on_epoch_end(self):
        # updates indices after each epoch
        self.index = np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.index)
