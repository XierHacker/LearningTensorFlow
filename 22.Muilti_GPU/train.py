import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import model

print(tf.__version__)

(x_train, y_train), (x_test, y_test)=tf.keras.datasets.fashion_mnist.load_data()

#expand dim to use convlution 2D
x_train=np.expand_dims(a=x_train,axis=-1)/np.float32(255)
# print("x_train:\n",x_train)

x_test=np.expand_dims(a=x_test,axis=-1)/np.float32(255)
# print("x_test:\n",x_test)

#
# plt.imshow(X=x_train[1,:,:,0])
# plt.show()


def train_step():
    pass

def train():
    pass



if __name__=="__main__":
    pass
