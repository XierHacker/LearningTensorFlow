import tensorflow as tf
import numpy as np
a=np.array([[1,2],[3,4]])


x=tf.constant(value=a)
tailed=tf.tile(input=x,multiples=[1,2])
tile=tf.reshape(tensor=tailed,shape=(-1,2))
#concated=tf.concat(values=[x,x],axis=0)
#transed=tf.transpose(concated,perm=())

with tf.Session() as sess:
    print(sess.run(x))
    print(sess.run(tailed))
    print(sess.run(tile))