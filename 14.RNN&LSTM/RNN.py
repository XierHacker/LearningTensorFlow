import tensorflow as tf
import numpy as np


'''
cell = tf.nn.rnn_cell.BasicRNNCell(num_units=128) # state_size = 128
print(cell.state_size) # 128

inputs = tf.placeholder(np.float32, shape=(32, 100)) # 32 是 batch_size
h0 = cell.zero_state(32, np.float32) # 通过zero_state得到一个全0的初始状态，形状为(batch_size, state_size)
output, h1 = cell.call(inputs, h0) #调用call函数

print(h1.shape) # (32, 128)
print(output.shape)

lstm_cell_forward = tf.nn.rnn_cell.BasicLSTMCell(num_units=128)
lstm_cell_backward = tf.nn.rnn_cell.BasicLSTMCell(num_units=128)
inputs = tf.placeholder(np.float32, shape=(32, 100)) # 32 是 batch_size
lstm_cell=tf.nn.bidirectional_dynamic_rnn(
    cell_fw=lstm_cell_forward,
    cell_bw=lstm_cell_backward,
    inputs=inputs,
)
h0 = lstm_cell.zero_state(32, np.float32) # 通过zero_state得到一个全0的初始状态
print("shape of h0:",h0.shape)
output, h1 = lstm_cell.call(inputs, h0)

print(h1.h)  # shape=(32, 128)
print(h1.c)  # shape=(32, 128)

'''



'''
inputs = tf.placeholder(np.float32, shape=(32, 100)) # 32 是 batch_size
h0 = lstm_cell.zero_state(32, np.float32) # 通过zero_state得到一个全0的初始状态
print("len of h0:",len(h0))
print("shape of h0.h:",h0.h.shape)
print("shape of h0.c:",h0.c.shape)
output, h1 = lstm_cell.call(inputs,h0)

print(h1.h)  # shape=(32, 128)
print(h1.c)  # shape=(32, 128)


import tensorflow as tf
import numpy as np
lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(num_units=128)
lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(num_units=256)
lstm_cell_3 = tf.contrib.rnn.BasicLSTMCell(num_units=512)
#多层lstm_cell
lstm_cell=tf.contrib.rnn.MultiRNNCell(cells=[lstm_cell_1,lstm_cell_2,lstm_cell_3])

print("output_size:",lstm_cell.output_size)
print("state_size:",lstm_cell.state_size)
#print(lstm_cell.state_size.h)
#print(lstm_cell.state_size.c)
'''

import tensorflow as tf
import numpy as np

inputs = tf.placeholder(np.float32, shape=(32,40,5)) # 32 是 batch_size
lstm_cell_fw = tf.contrib.rnn.BasicLSTMCell(num_units=128)
lstm_cell_bw = tf.contrib.rnn.BasicLSTMCell(num_units=128)

#多层lstm_cell
#lstm_cell=tf.contrib.rnn.MultiRNNCell(cells=[lstm_cell_1,lstm_cell_2,lstm_cell_3])

print("output_fw_size:",lstm_cell_fw.output_size)
print("state_fw_size:",lstm_cell_fw.state_size)
print("output_bw_size:",lstm_cell_bw.output_size)
print("state_bw_size:",lstm_cell_bw.state_size)

#print(lstm_cell.state_size.h)
#print(lstm_cell.state_size.c)
output,state=tf.nn.bidirectional_dynamic_rnn(
    cell_fw=lstm_cell_fw,
    cell_bw=lstm_cell_bw,
    inputs=inputs,
    dtype=tf.float32
)
output_fw=output[0]
output_bw=output[1]
state_fw=state[0]
state_bw=state[1]

print("output_fw.shape:",output_fw.shape)
print("output_bw.shape:",output_bw.shape)
print("len of state tuple",len(state_fw))
print("state_fw:",state_fw)
print("state_bw:",state_bw)
#print("state.h.shape:",state.h.shape)
#print("state.c.shape:",state.c.shape)

#state_concat=tf.concat(values=[state_fw,state_fw],axis=1)
#print(state_concat)
state_h_concat=tf.concat(values=[state_fw.h,state_bw.h],axis=1)
print("state_fw_h_concat.shape",state_h_concat.shape)

state_c_concat=tf.concat(values=[state_fw.c,state_bw.c],axis=1)
print("state_fw_h_concat.shape",state_c_concat.shape)

state_concat=tf.contrib.rnn.LSTMStateTuple(c=state_c_concat,h=state_h_concat)
print(state_concat)