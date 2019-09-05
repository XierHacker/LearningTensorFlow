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


inceptionv3 = tf.keras.applications.InceptionV3(include_top=False,weights='imagenet')
new_input = inceptionv3.input
hidden_layer = inceptionv3.layers[-1].output

image_features_extract_model = tf.keras.Model(new_input, hidden_layer)