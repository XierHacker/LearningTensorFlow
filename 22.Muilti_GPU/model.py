import os
import numpy as np
import tensorflow as tf

class CNN(tf.keras.Model):
    def __init__(self,filters,kernel_size,strides,activation):
        super(CNN,self).__init__()
        self.conv1=tf.keras.layers.Conv2D(filters=filters,kernel_size=kernel_size,strides=strides,activation=activation)
        self.max_pool1=tf.keras.layers.MaxPooling2D()
        self.conv2=tf.keras.layers.Conv2D(filters=filters,kernel_size=kernel_size,strides=strides,activation=activation)
        self.max_pool2=tf.keras.layers.MaxPooling2D()
        self.flatten=tf.keras.layers.Flatten()
        self.fc1=tf.keras.layers.Dense(units=filters,activation=activation)
        self.fc2=tf.keras.layers.Dense(units=10,activation=None)
    
    def __call__(self,x):
        x=self.conv1(x)
        x=self.max_pool1(x)
        x=self.conv2(x)
        x=self.max_pool2(x)
        x=self.flatten(x)
        x=self.fc1(x)
        x=self.fc2(x)
        #print("x.shape",x)
        return x




if __name__=="__main__":
    cnn=CNN(filters=64,kernel_size=3,strides=1,activation=tf.nn.relu)

    inputs=tf.ones(shape=(30,28,28,3),dtype=tf.float32)

    cnn(inputs)
    
