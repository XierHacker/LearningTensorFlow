'''
__init__(
    num_units,
    use_peepholes=False,
    cell_clip=None,
    initializer=None,
    num_proj=None,
    proj_clip=None,
    num_unit_shards=None,
    num_proj_shards=None,
    forget_bias=1.0,
    state_is_tuple=True,
    activation=None,
    reuse=None,
    name=None,
    dtype=None
)


Args:
num_units: LSTM cell的数量
use_peepholes: bool, 表示是否使用diagonal/peephole connections.
cell_clip: (optional) A float value, if provided the cell state is clipped by this value prior to the cell output activation.
initializer: (optional) The initializer to use for the weight and projection matrices.
num_proj: (optional) int, The output dimensionality for the projection matrices. If None, no projection is performed.
proj_clip: (optional) A float value. If num_proj > 0 and proj_clip is provided, then the projected values are clipped elementwise to within [-proj_clip, proj_clip].
num_unit_shards: Deprecated, will be removed by Jan. 2017. Use a variable_scope partitioner instead.
num_proj_shards: Deprecated, will be removed by Jan. 2017. Use a variable_scope partitioner instead.
forget_bias: Biases of the forget gate are initialized by default to 1 in order to reduce the scale of forgetting at the beginning of the training. Must set it manually to 0.0 when restoring from CudnnLSTM trained checkpoints.
state_is_tuple: If True, accepted and returned states are 2-tuples of the c_state and m_state. If False, they are concatenated along the column axis. This latter behavior will soon be deprecated.
activation: Activation function of the inner states. Default: tanh.
reuse: (optional) Python boolean describing whether to reuse variables in an existing scope. If not True, and the existing scope already has the given variables, an error is raised.
name: String, the name of the layer. Layers with the same name will share weights, but to avoid mistakes we require reuse=True in such cases.
dtype: Default dtype of the layer (default of None means use the type of the first input). Required when build is called before call.

When restoring from CudnnLSTM-trained checkpoints, use CudnnCompatibleLSTMCell instead.
'''



import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers

lstm_cell_1=tf.nn.rnn_cell.LSTMCell(
    num_units=128,
    use_peepholes=True,
    initializer=initializers.xavier_initializer(),
    num_proj=64,
    name="LSTM_CELL_1"
)

lstm_cell_2=tf.nn.rnn_cell.LSTMCell(
    num_units=128,
    use_peepholes=True,
    initializer=initializers.xavier_initializer(),
    num_proj=64,
    name="LSTM_CELL_2"
)


lstm_cell_3=tf.nn.rnn_cell.LSTMCell(
    num_units=128,
    use_peepholes=True,
    initializer=initializers.xavier_initializer(),
    num_proj=64,
    name="LSTM_CELL_3"
)

multi_cell=tf.nn.rnn_cell.MultiRNNCell(cells=[lstm_cell_1,lstm_cell_2,lstm_cell_3])

X=tf.ones(shape=(20,30,128),dtype=tf.float32)

outputs,states=tf.nn.dynamic_rnn(cell=multi_cell,inputs=X,dtype=tf.float32)

print("outputs:",outputs)
print("states:",states)

# print("output_size:",multi_cell.output_size)
# print("state_size:",type(multi_cell.state_size))
# print("state_size:",multi_cell.state_size)
#
# #需要先索引到具体的那层cell，然后取出具体的state状态
# print(multi_cell.state_size[0].h)
# print(multi_cell.state_size[0].c)

