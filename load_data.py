import os

import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from config import inputsize, train_data, train_split
import numpy as np
import time
import utils
from augument import get_more_images_v1, get_more_images_v3

def one_hot_encode(y):
    # one hot encode outputs
    y = np_utils.to_categorical(y)
    num_classes = y.shape[1]
    return y, num_classes


def load_datasets(dataset=train_data, inputsize=inputsize):
    X = []
    y = []
    label = os.listdir(dataset)
    for image_label in label:
        images = os.listdir(os.path.join(dataset, image_label))
        for image in tqdm(images):
            path = os.path.join(dataset + image_label + '/', image)
            img = cv2.imread(path)
            if img is not None:
                img = cv2.resize(img, inputsize)
                img = np.asarray(img)
                X.append(img)
                y.append(label.index(image_label))

    X = np.asarray(X)
    y = np.asarray(y)
    # one hot encode
    y, num_class = one_hot_encode(y)

    # create more images for train

    n_generate_times = 10
    X = get_more_images_v3(X, n_generate_times, 2)
    result = y
    for i in range(1, n_generate_times):
        result = np.concatenate((result, y))
    y = result
    return X, y, num_class

def get_data_size(dataset):
    total = 0
    label = os.listdir(dataset)
    for image_label in label:
        images = os.listdir(os.path.join(dataset, image_label))
        total += len(images)
    return total