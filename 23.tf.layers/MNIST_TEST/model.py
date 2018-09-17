import tensorflow as tf
import numpy as np
import parameter
from tensorflow.contrib.layers import xavier_initializer

class Model():
    def __init__(self):
        self.class_num=10

    def forward(self,input,regularizer):
        #---------------------------------------conv1--------------------------------------#
        logits_conv1=tf.layers.conv2d(
            inputs=input,filters=32,kernel_size=5,strides=1,padding="SAME",activation=tf.nn.relu,
            kernel_initializer=xavier_initializer(),kernel_regularizer=regularizer,name="conv_1"
        )
        print("shape of logits_conv1:",logits_conv1.shape)
        logits_conv1=tf.layers.max_pooling2d(inputs=logits_conv1,pool_size=2,strides=2,padding="SAME",name="padded_conv1")
        print("shape of logits_conv1:", logits_conv1.shape)

        # ---------------------------------------conv2--------------------------------------#
        logits_conv2 = tf.layers.conv2d(
            inputs=logits_conv1, filters=64, kernel_size=5, strides=1, padding="SAME", activation=tf.nn.relu,
            kernel_initializer=xavier_initializer(),kernel_regularizer=regularizer, name="conv_2"
        )
        print("shape of logits_conv2:", logits_conv2.shape)
        logits_conv2 = tf.layers.max_pooling2d(inputs=logits_conv2, pool_size=2, strides=2, padding="SAME",name="padded_conv2")
        print("shape of logits_conv2:", logits_conv2.shape)

        logits_flatten=tf.layers.flatten(inputs=logits_conv2,name="logits_flatten")
        print("shape of logits_flatten:", logits_flatten.shape)

        logits=tf.layers.dense(inputs=logits_flatten,units=10,activation=tf.nn.relu,
                               kernel_initializer=xavier_initializer(),
                               kernel_regularizer=regularizer,
                               name="logits")
        print("shape of logits:", logits.shape)
        return logits


