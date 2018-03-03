import tensorflow as tf
import numpy as np

array=np.arange(start=0,stop=90)
#print(a)
a=array.reshape((10,9))
#print(a)

b=array.reshape((10,3,3))
print(b)
lens=np.array([2,2,2,2,2,2,2,2,2,2])

#lens_b=np.array([[2,3],[2,3],[2,3],[2,3],[2,3],[2,3],[2,3],[2,3],[2,3],[2,3]])



X_p=tf.placeholder(dtype=tf.int32,shape=(None,3,3))
sequence_len_p=tf.placeholder(dtype=tf.int32,shape=(None,))
mask=tf.sequence_mask(lengths=sequence_len_p,maxlen=3)
x=X_p
x_masked=tf.boolean_mask(tensor=X_p,mask=mask)


with tf.Session() as sess:
    '''
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

    '''

    sequence_len, mask_index_get, X,X_masked = sess.run(
        fetches=[sequence_len_p, mask, x,x_masked],
        feed_dict={
            X_p: b[:5],
            sequence_len_p: lens[:5]
        }
    )

    print("sequence_len:", sequence_len)
    print("mask_index_get:\n", mask_index_get.shape)
    print("X:\n", X)
    print("X_masked:\n", X_masked)
