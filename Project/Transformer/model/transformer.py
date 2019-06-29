import os
import sys
sys.path.append("../")
import numpy as np 
import tensorflow as tf 
from utils.display import draw_positional_encodings


def positional_encoding(max_pos,d_model):
    '''
        《Attention Is All You Need》 3.5
    '''
    #method 2
    pos=np.arange(max_pos)[:,np.newaxis]
    i=np.arange(d_model)[np.newaxis, :]
    #compute whole PE
    PE=(pos*1.0)/np.power(10000, (2 * (i//2)) / np.float32(d_model))
    # apply sin to even indices in the array; 2i
    sines = np.sin(PE[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    cosines = np.cos(PE[:, 1::2])
    # concat
    PE = np.concatenate([sines, cosines], axis=-1)
    PE = PE[np.newaxis, ...]
    return tf.cast(PE, dtype=tf.float32)


def padding_mask(seq):
    '''
        sep should a [batch_size,max_size]
    '''

    mask=tf.cast(tf.math.equal(seq,0),tf.float32)
    print("mask:\n",mask)


def look_ahead_mask(size):
    pass


def scaled_dot_product_attention(q, k, v, mask):
    '''
        《Attention Is All You Need》 3.2.1

        q, k, v must have matching leading dimensions.
        k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
        The mask has different shapes depending on its type(padding or look ahead) 
        but it must be broadcastable for addition.
  
        Args:
            q: query shape == (..., seq_len_q, depth)
            k: key shape == (..., seq_len_k, depth)
            v: value shape == (..., seq_len_v, depth_v)
            mask: Float tensor with shape broadcastable 
                to (..., seq_len_q, seq_len_k). Defaults to None.
    
        Returns:
            output, attention_weights
    '''

















    



if __name__=="__main__":
    pos_encoding = positional_encoding(50, 512)
    print (pos_encoding.shape)
    #draw_positional_encodings(pos_encoding)

    padding_mask(seq=[[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])