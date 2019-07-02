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









# pt_batch, en_batch = next(iter(train_dataset))
# print("pt_batch:\n",pt_batch)
# print("en_batch:\n",en_batch)


def create_padding_mask(seq):
    '''
        sep should a [batch_size,max_size]
    '''
    mask=tf.cast(tf.math.equal(seq,0),tf.float32)
    # add extra dimensions so that we can add the padding to the attention logits.
    return mask[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)
    


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)

def create_masks(inp, tar):
  # Encoder padding mask
  enc_padding_mask = create_padding_mask(inp)
  
  # Used in the 2nd attention block in the decoder.
  # This padding mask is used to mask the encoder outputs.
  dec_padding_mask = create_padding_mask(inp)
  
  # Used in the 1st attention block in the decoder.
  # It is used to pad and mask future tokens in the input received by 
  # the decoder.
  look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
  dec_target_padding_mask = create_padding_mask(tar)
  #combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
  
  #return enc_padding_mask, combined_mask, dec_padding_mask
  return enc_padding_mask, dec_padding_mask



train_dataset=process.get_dataset()
# cache the dataset to memory to get a speedup while reading from it.
train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes=([-1], [-1]))
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

num=0
for (batch, (inp, tar)) in enumerate(train_dataset):
    if num>2:
        break
    print("batch:",batch)
    print("inp:\n",inp)

    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    print("tar:\n",tar)
    print("tar_inp:\n",tar_inp)
    print("tar_real:\n",tar_real)
    print("\n\n")

    enc_padding_mask, dec_padding_mask=create_masks(inp,tar)
    print("enc_padding_mask:\n",enc_padding_mask)

    num+=1




    


if __name__=="__main__":
    print("start")