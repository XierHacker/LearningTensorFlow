import os
import sys
sys.path.append("../")
import numpy as np 
import tensorflow as tf 
import tensorflow_datasets as tfds 
from transformer import *
from dataset import process

#load subword and restore tokenizer
tokenizer_en=tfds.features.text.SubwordTextEncoder.load_from_file(filename_prefix="../index_files/en")
tokenizer_pt=tfds.features.text.SubwordTextEncoder.load_from_file(filename_prefix="../index_files/pt")


BUFFER_SIZE = 20000
BATCH_SIZE = 64

source_vocab_size=tokenizer_pt.vocab_size+2
target_vocab_size=tokenizer_en.vocab_size+2
print("tokenizer_en.size:",target_vocab_size)
print("tokenizer_pt.size:",source_vocab_size)

NUM_LAYERS=4
D_MODEL=128
DFF=512
NUM_HEADS=8
DROPOUT_RATE=0.1






# train_dataset=process.get_dataset()
# # cache the dataset to memory to get a speedup while reading from it.
# train_dataset = train_dataset.cache()
# train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes=([-1], [-1]))
# train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

# pt_batch, en_batch = next(iter(train_dataset))
# print("pt_batch:\n",pt_batch)
# print("en_batch:\n",en_batch)


def create_padding_mask(seq):
    '''
        sep should a [batch_size,max_size]
    '''
    mask=tf.cast(tf.math.equal(seq,0),tf.float32)
    mask=mask * -1e9
    print("mask:\n",mask)


def create_look_ahead_mask(size):
    pass




if __name__=="__main__":
    pass