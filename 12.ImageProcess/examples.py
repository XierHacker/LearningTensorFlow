import os
import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf

#read picture
pic=mpimg.imread("../../data/DogsVsCats/train/cat.0.jpg")
print(pic)
plt.imshow(pic)
plt.show()

#randon random_flip_left_right
flip_pic=tf.image.random_flip_left_right(image=pic)

#tf.image.random_flip_up_down
flip_pic=tf.image.random_flip_up_down(image=flip_pic)

new=tf.image.resize_image_with_crop_or_pad(image=pic,target_height=224,target_width=224)

with tf.Session() as sess:
    plt.imshow(sess.run(flip_pic))
    plt.show()