import numpy as np
import tensorflow as tf 
import tensorflow_datasets as tfds 

print(tfds.list_builders())

examples,info=tfds.load(name="mnist",with_info=True,as_supervised=True)

print("examples:\n",examples)
print("info:\n",info)