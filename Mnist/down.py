import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import pandas

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
print(mnist.train.images.shape)

print(mnist.train.images[0].shape)

batch = mnist.train.next_batch(10)


print(batch[0].shape)
print(batch[1].shape)
print()