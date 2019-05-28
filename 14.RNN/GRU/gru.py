'''
tf.contrib.rnn.GRUCell

__init__(
    num_units,
    activation=None,
    reuse=None,
    kernel_initializer=None,
    bias_initializer=None,
    name=None,
    dtype=None
)

Args:
num_units: int, GRU cell 中的units数量
activation: Nonlinearity to use. Default: tanh.
reuse: (optional) Python boolean describing whether to reuse variables in an existing scope. If not True, and the existing scope already has the given variables, an error is raised.
kernel_initializer: (optional) The initializer to use for the weight and projection matrices.
bias_initializer: (optional) The initializer to use for the bias.
name: String, the name of the layer. Layers with the same name will share weights, but to avoid mistakes we require reuse=True in such cases.
dtype: Default dtype of the layer (default of None means use the type of the first input). Required when build is called before call.

'''


import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers

gru_cell=tf.nn.rnn_cell.GRUCell(
    num_units=128,
    kernel_initializer=initializers.xavier_initializer(),
    bias_initializer=tf.initializers.truncated_normal()
)


#gru_cell=tf.nn.rnn_cell.LSTMCell(
#    num_units=128,
#    use_peepholes=True,
#    initializer=initializers.xavier_initializer(),
#    num_proj=64,
#    name="LSTM_CELL"
#)

print("output_size:",gru_cell.output_size)      #output_size: 128
print("state_size:",gru_cell.state_size)        #state_size: 128