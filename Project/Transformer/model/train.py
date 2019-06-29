import os
import sys
sys.path.append("../")
import numpy as np 
import tensorflow as tf 
from dataset import process



BUFFER_SIZE = 20000
BATCH_SIZE = 64

train_dataset=process.get_dataset()
# cache the dataset to memory to get a speedup while reading from it.
train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes=([-1], [-1]))
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

pt_batch, en_batch = next(iter(train_dataset))
print("pt_batch:\n",pt_batch)
print("en_batch:\n",en_batch)
