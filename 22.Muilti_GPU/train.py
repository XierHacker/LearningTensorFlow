import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import model


DEVICES_LIST=["gpu:4","gpu:5","gpu:6"]
#parameters
BATCH_SIZE=2
CLASS_NUM=10
EPOCHS=10





print(tf.__version__)

(x_train, y_train), (x_test, y_test)=tf.keras.datasets.fashion_mnist.load_data()

#expand dim to use convlution 2D
x_train=np.expand_dims(a=x_train,axis=-1)/np.float32(255)
train_dataset=tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(BATCH_SIZE)
# for records in train_dataset:
#     print("records:\n",records[0])
#     print("records:\n",records[1])
    

x_test=np.expand_dims(a=x_test,axis=-1)/np.float32(255)
test_dataset=tf.data.Dataset.from_tensor_slices((x_test,y_test))

#
# plt.imshow(X=x_train[1,:,:,0])
# plt.show()


def train_step():
    pass

def train():
    strategy=tf.distribute.MirroredStrategy(devices=DEVICES_LIST)
    print("num devices:",strategy.num_replicas_in_sync)
    with strategy.scope():
        train_dist_dataset=strategy.experimental_distribute_dataset(train_dataset)
        for records in train_dist_dataset:
            print("records:\n",records[0])
            print("records:\n",records[1])
            


    



if __name__=="__main__":
    train()

