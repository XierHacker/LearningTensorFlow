'''
__init__
__init__(
    num_units,
    forget_bias=1.0,
    activation=None,
    reuse=None,
    kernel_initializer=None,
    bias_initializer=None,
    name=None,
    dtype=None
)
Initialize the IndyLSTM cell.

Args:
num_units: int, The number of units in the LSTM cell.
forget_bias: float, The bias added to forget gates (see above). Must set to 0.0 manually when restoring from CudnnLSTM-trained checkpoints.
activation: Activation function of the inner states. Default: tanh.
reuse: (optional) Python boolean describing whether to reuse variables in an existing scope. If not True, and the existing scope already has the given variables, an error is raised.
kernel_initializer: (optional) The initializer to use for the weight matrix applied to the inputs.
bias_initializer: (optional) The initializer to use for the bias.
name: String, the name of the layer. Layers with the same name will share weights, but to avoid mistakes we require reuse=True in such cases.
dtype: Default dtype of the layer (default of None means use the type of the first input). Required when build is called before call.

'''

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers
import tensorflow.contrib.rnn as rnn

indy_lstm_cell=rnn.IndyLSTMCell(num_units=128)


#lstm_cell=tf.nn.rnn_cell.LSTMCell(
#    num_units=128,
#    use_peepholes=True,
#    initializer=initializers.xavier_initializer(),
#    num_proj=64,
#    name="LSTM_CELL"
#)

print("output_size:",indy_lstm_cell.output_size)
print("state_size:",indy_lstm_cell.state_size)
print(indy_lstm_cell.state_size.h)
print(indy_lstm_cell.state_size.c)