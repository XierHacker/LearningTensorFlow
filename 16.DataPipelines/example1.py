import tensorflow as tf
import pandas as pd
import numpy as np



#-------------------------------Generate Data--------------------------------------#
#generate data
train_frame = pd.read_csv("../Mnist/train.csv")
test_frame = pd.read_csv("../Mnist/test.csv")

# pop the labels and one-hot coding
train_labels_frame = train_frame.pop("label")

# get values
# one-hot on labels
X_train = train_frame.astype(np.float32).values
y_train=pd.get_dummies(data=train_labels_frame).values
X_test = test_frame.astype(np.float32).values

#trans the shape to (batch,time_steps,input_size)
X_train=np.reshape(X_train,newshape=(-1,28,28))
X_test=np.reshape(X_test,newshape=(-1,28,28))
print(X_train.shape)
print(y_train.shape)
#print(X_test.shape)

#-----------------------------------------------------------------------------------#

#create dataset
dataset=tf.data.Dataset.from_tensor_slices(tensors=(X_train,y_train))
print(dataset.output_shapes)
#print(dataset[0])

#create iterator
iterator=dataset.make_one_shot_iterator()
elememt=iterator.get_next()

with tf.Session() as sess:
    ele=sess.run(elememt)
    print("image:\n",ele[0])
    print("label:\n",ele[1])

