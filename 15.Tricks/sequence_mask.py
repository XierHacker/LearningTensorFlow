import tensorflow as tf
import numpy as np

a=np.arange(start=0,stop=50)
#print(a)
a=a.reshape((10,5))
#print(a)
lens=np.array([2,2,2,2,2,2,2,2,2,2])


X_p=tf.placeholder(dtype=tf.int32,shape=(None,5))
sequence_len_p=tf.placeholder(dtype=tf.int32,shape=(None,))
mask=tf.sequence_mask(lengths=sequence_len_p,maxlen=5)
x=X_p
x_masked=tf.boolean_mask(tensor=X_p,mask=mask)


with tf.Session() as sess:
    sequence_len,mask_index_get,X,X_masked=sess.run(
        fetches=[sequence_len_p,mask,x,x_masked],
        feed_dict={
            X_p:a[:5],
            sequence_len_p:lens[:5]
        }
    )
    print("sequence_len:",sequence_len)
    print("mask_index_get:\n",mask_index_get)
    print("X:\n",X)
    print("X_masked:",X_masked)
