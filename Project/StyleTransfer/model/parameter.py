import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


content_path = tf.keras.utils.get_file(
  'turtle.jpg',
  'https://storage.googleapis.com/download.tensorflow.org/example_images/Green_Sea_Turtle_grazing_seagrass.jpg'
)
style_path = tf.keras.utils.get_file(
  'kandinsky.jpg',
  'https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg'
)