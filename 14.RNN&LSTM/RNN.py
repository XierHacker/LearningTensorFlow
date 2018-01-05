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
'''



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