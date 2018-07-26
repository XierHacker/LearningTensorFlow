import os
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

N_GPU=2

a=np.arange(start=0,stop=100)
print("a:\n",a)

dataset=tf.data.Dataset.from_tensor_slices(tensors=a)

iterator=dataset.make_one_shot_iterator()
batch=iterator.get_next()

data_list=[]
for i in range(N_GPU):
    with tf.device("/gpu:%d" % i):
        with tf.name_scope("gpu_%d" % i) as scope:
            data=batch
            data_list.append(data)

with tf.Session() as sess:
    print("elements of dataset:")
    for i in range(2):
        #print(sess.run(data))
        print(data_list)
        print(sess.run(data_list[0]))
        print(sess.run(data_list[1]))
