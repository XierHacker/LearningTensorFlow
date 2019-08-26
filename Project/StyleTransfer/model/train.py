import os
import sys
sys.path.append("../")
sys.path.append("../../")
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


import parameter
import image_utils
import style_transfer






if __name__=="__main__":
  content_image = image_utils.load_img(parameter.content_path)
  style_image = image_utils.load_img(parameter.style_path)

  plt.subplot(1, 2, 1)
  image_utils.imshow(content_image, 'Content Image')

  plt.subplot(1, 2, 2)
  image_utils.imshow(style_image, 'Style Image')

  plt.show()