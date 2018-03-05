import tensorflow as tf
import numpy as np

a=np.array([1,2,3,4,5,6,7,8])

index=tf.constant(value=a,name="index")
#print(index)
one_hoted1=tf.one_hot(indices=index,depth=3)
print(one_hoted1.shape)

index2=tf.reshape(tensor=index,shape=[4,2],name="index2")
one_hoted2=tf.one_hot(indices=index2,depth=9)
print(one_hoted2.shape)

with tf.Session() as sess:
    print(sess.run(index))
    print(sess.run(one_hoted1))
    print(sess.run(one_hoted2))