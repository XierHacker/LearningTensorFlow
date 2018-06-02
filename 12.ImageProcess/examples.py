import os
import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf

pic=mpimg.imread("../../data/DogsVsCats/train/cat.0.jpg")
print(pic)
#plt.imshow(pic)
#plt.show()
cv2.imshow(winname="pic",mat=pic)
cv2.waitKey()

pic=cv2.resize(src=pic,dsize=(300,300),interpolation=cv2.INTER_AREA)
cv2.imshow(winname="pic",mat=pic)
cv2.waitKey()

new=tf.image.resize_image_with_crop_or_pad(image=pic,target_height=224,target_width=224)

with tf.Session() as sess:
    plt.imshow(sess.run(new))
    plt.show()