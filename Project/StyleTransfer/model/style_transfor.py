import os
import sys
import time
import tensorflow as tf

vgg=tf.keras.applications.VGG19(include_top=True, weights='imagenet')
