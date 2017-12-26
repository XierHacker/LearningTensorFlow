import tensorflow as tf
import numpy as np

img = tf.constant(value=[[[[1],[2],[3],[4]],[[1],[2],[3],[4]],[[1],[2],[3],[4]],[[1],[2],[3],[4]]]],dtype=tf.float32)
img = tf.concat(values=[img,img],axis=3)
print("img:",img)
filter = tf.constant(value=1, shape=[3,3,2,5], dtype=tf.float32)
out_img1 = tf.nn.atrous_conv2d(value=img, filters=filter, rate=1, padding='SAME')
out_img2 = tf.nn.atrous_conv2d(value=img, filters=filter, rate=1, padding='VALID')
out_img3 = tf.nn.atrous_conv2d(value=img, filters=filter, rate=2, padding='SAME')

#error
#out_img4 = tf.nn.atrous_conv2d(value=img, filters=filter, rate=2, padding='VALID')

with tf.Session() as sess:
    print("original img:")
    print(sess.run(img[0]))
    print('rate=1, SAME mode result:')
    print(sess.run(out_img1))

    print('rate=1, VALID mode result:')
    print(sess.run(out_img2))

    print('rate=2, SAME mode result:')
    print(sess.run(out_img3))

    # error
    #print 'rate=2, VALID mode result:'
    #print(sess.run(out_img4))