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