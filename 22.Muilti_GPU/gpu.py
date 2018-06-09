import numpy as np
import tensorflow as tf

config=tf.ConfigProto(log_device_placement=True)

a=tf.ones(shape=(3,3),dtype=tf.int32,name="a")
b=tf.ones(shape=(3,3),dtype=tf.int32,name="b")

c=tf.add(a,b,name="c")

with tf.Session(config=config) as sess:
    print(sess.run(c))