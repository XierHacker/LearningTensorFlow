import numpy as np
import os
import time
from PIL import Image
import tensorflow as tf


inceptionv3 = tf.keras.applications.InceptionV3(include_top=False,weights='imagenet')
new_input = inceptionv3.input
hidden_layer = inceptionv3.layers[-1].output

image_features_extract_model = tf.keras.Model(new_input, hidden_layer)


