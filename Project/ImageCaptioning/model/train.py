import re
import numpy as np
import os
import time
import json
from glob import glob
from PIL import Image
import pickle
from sklearn.utils import shuffle
import tensorflow as tf

import model
import preprocess


def extract_img_features(img_names,extractor):
    # Get unique images
    encode_train = sorted(set(img_names))
    print("encode_train:\n",encode_train)

    # # Feel free to change batch_size according to your system configuration
    # image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
    # image_dataset = image_dataset.map(
    #   load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)
    #
    # for img, path in image_dataset:
    #   batch_features = image_features_extract_model(img)
    #   batch_features = tf.reshape(batch_features,
    #                               (batch_features.shape[0], -1, batch_features.shape[3]))
    #
    #   for bf, p in zip(batch_features, path):
    #     path_of_feature = p.numpy().decode("utf-8")
    #     np.save(path_of_feature, bf.numpy())

if __name__=="__main__":
    train_captions, train_img_names = get_image_annotation()
    extract_img_features(img_names=train_img_names,extractor=model.image_features_extract_model)
