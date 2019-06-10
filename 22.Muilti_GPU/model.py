import os
import numpy as np
import tensorflow as tf

class CNN(tf.keras.Model):
    def __init__(self,filters,kernel_size,strides,activation):
        super(CNN,self).__init__()
        conv1=tf.keras.layers.Conv2D(filters=filters,kernel_size=kernel_size,strides=strides,activation=activation)
        max_pool1=tf.keras.layers.MaxPooling2D()
        
    
    def __call__():
        pass

