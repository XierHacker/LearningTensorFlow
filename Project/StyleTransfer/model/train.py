import os
import sys
sys.path.append("../")
sys.path.append("../../")
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import tensorflow as tf

import parameter
import image_utils
from style_transfer import StyleContentModel


def clip_0_1(image):
  return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


def train(content_image,style_image):
  extractor = StyleContentModel(parameter.style_layers, parameter.content_layers)

  style_targets = extractor(style_image)['style']
  content_targets = extractor(content_image)['content']

  image = tf.Variable(content_image)

  opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

  style_weight = 1e-2
  content_weight = 1e4

  def style_content_loss(outputs):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2)
                           for name in style_outputs.keys()])
    style_loss *= style_weight / parameter.num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2)
                             for name in content_outputs.keys()])
    content_loss *= content_weight / parameter.num_content_layers
    loss = style_loss + content_loss
    return loss

  for i in range(100):
    start_time=time.time()
    print("Step:",i)
    with tf.GradientTape() as tape:
      outputs = extractor(image)
      loss = style_content_loss(outputs)
    print("---loss:",loss.numpy())
    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))
    end_time=time.time()
    print("---spend:",end_time-start_time," seconds")


  #save result
  img.imsave("./result.png",image[0])

  #plt.imshow(image.read_value()[0])
  #plt.show()







if __name__=="__main__":
  content_image = image_utils.load_img(parameter.content_path)
  style_image = image_utils.load_img(parameter.style_path)
  train(content_image=content_image,style_image=style_image)
  #
  # plt.subplot(1, 2, 1)
  # image_utils.imshow(content_image, 'Content Image')
  #
  # plt.subplot(1, 2, 2)
  # image_utils.imshow(style_image, 'Style Image')
  #
  # plt.show()

  # x = tf.keras.applications.vgg19.preprocess_input(content_image*255)
  # #print("x:\n",x)
  # x = tf.image.resize(x, (224, 224))
  # #print("x:\n", x)
  # vgg = tf.keras.applications.VGG19(include_top=True, weights='imagenet')
  # prediction_probabilities = vgg(x)
  # print("pred:\n",prediction_probabilities.shape)
  #
  # predicted_top_5 = tf.keras.applications.vgg19.decode_predictions(prediction_probabilities.numpy())[0]
  # result=[(class_name, prob) for (number, class_name, prob) in predicted_top_5]
  # print("result:\n",result)

  # style_outputs = style_extractor(style_image * 255)
  # for name, output in zip(style_layers, style_outputs):
  #   print(name)
  #   print("  shape: ", output.numpy().shape)
  #   print("  min: ", output.numpy().min())
  #   print("  max: ", output.numpy().max())
  #   print("  mean: ", output.numpy().mean())
  #   print()


  # results = extractor(tf.constant(content_image))
  #
  # style_results = results['style']
  #
  # print('Styles:')
  # for name, output in sorted(results['style'].items()):
  #   print("  ", name)
  #   print("    shape: ", output.numpy().shape)
  #   print("    min: ", output.numpy().min())
  #   print("    max: ", output.numpy().max())
  #   print("    mean: ", output.numpy().mean())
  #   print()
  #
  # print("Contents:")
  # for name, output in sorted(results['content'].items()):
  #   print("  ", name)
  #   print("    shape: ", output.numpy().shape)
  #   print("    min: ", output.numpy().min())
  #   print("    max: ", output.numpy().max())
  #   print("    mean: ", output.numpy().mean())
