'''
tf.keras.layers.LSTM


__init__(
    units,
    activation='tanh',
    recurrent_activation='sigmoid',
    use_bias=True,
    kernel_initializer='glorot_uniform',
    recurrent_initializer='orthogonal',
    bias_initializer='zeros',
    unit_forget_bias=True,
    kernel_regularizer=None,
    recurrent_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    recurrent_constraint=None,
    bias_constraint=None,
    dropout=0.0,
    recurrent_dropout=0.0,
    implementation=1,
    return_sequences=False,
    return_state=False,
    go_backwards=False,
    stateful=False,
    time_major=False,
    unroll=False,
    **kwargs
)


参数:
- units: 正整数，表示输出维度
- activation: 激活函数，默认是tanh，要是传入为None，那么不会有任何激活函数被使用
- recurrent_activation: 在递归步(recurrent step)使用的激活函数，默认是hard_sigmoid. 要是传入为None，那么不会有任何激活函数被使用
- use_bias: 布尔值，表示是否使用bias
- kernel_initializer: input的线性变换这里的kernel权重矩阵的初始化方法
- recurrent_initializer: recurent这里的线性变换这里的kernel权重矩阵的初始化方法
- bias_initializer: bias初始化方法
- unit_forget_bias: Boolean. If True, add 1 to the bias of the forget gate at initialization. Setting it to true will also force bias_initializer="zeros". This is recommended in Jozefowicz et al..
- kernel_regularizer: Regularizer function applied to the kernel weights matrix.
- recurrent_regularizer: Regularizer function applied to the recurrent_kernel weights matrix.
- bias_regularizer: Regularizer function applied to the bias vector.
- activity_regularizer: Regularizer function applied to the output of the layer (its "activation")..
- kernel_constraint: Constraint function applied to the kernel weights matrix.
- recurrent_constraint: Constraint function applied to the recurrent_kernel weights matrix.
- bias_constraint: Constraint function applied to the bias vector.
- dropout: Float between 0 and 1. Fraction of the units to drop for the linear transformation of the inputs.
- recurrent_dropout: Float between 0 and 1. Fraction of the units to drop for the linear transformation of the recurrent state.
- implementation: Implementation mode, either 1 or 2. Mode 1 will structure its operations as a larger number of smaller dot products and additions, whereas mode 2 will batch them into fewer, larger operations. These modes will have different performance profiles on different hardware and for different applications.
- return_sequences: 布尔值，表示是否返回最后一个输出或者是整个时间步上面的输出。
- return_state: 布尔值，表示是否返回最后的state。
- go_backwards: Boolean (default False). If True, process the input sequence backwards and return the reversed sequence.
- stateful: Boolean (default False). If True, the last state for each sample at index i in a batch will be used as initial state for the sample of index i in the following batch.
- unroll: Boolean (default False). If True, the network will be unrolled, else a symbolic loop will be used. Unrolling can speed-up a RNN, although it tends to be more memory-intensive. Unrolling is only suitable for short sequences.

调用参数:
- inputs: 一个3D的tensor，形状为（batch，time_step,dim）
- mask: Binary tensor of shape (samples, timesteps) indicating whether a given timestep should be masked.
- training: Python boolean indicating whether the layer should behave in training mode or in inference mode. This argument is passed to the cell when calling it. This is only relevant if dropout or recurrent_dropout is used.
- initial_state: List of initial state tensors to be passed to the first call of the cell.





tf.keras.layers.Bidirectional

作用：双向RNN的封装

参数:
- layer: 循环神经网络相关层的一些实例.
- merge_mode: RNN的前向和反向融合在一起的方式，可以是{'sum', 'mul', 'concat', 'ave', None}其中之一. 如果选择None，那么不会融合，返回一个list

调用参数:
和封装的RNN实例的调用参数一样


Examples:
model = Sequential()
model.add(Bidirectional(LSTM(10, return_sequences=True), input_shape=(5,
10)))
model.add(Bidirectional(LSTM(10)))
model.add(Dense(5))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

__init__
__init__(
    layer,
    merge_mode='concat',
    weights=None,
    **kwargs
)

Properties
constraints
Methods
reset_states
reset_states()


'''



# import numpy as np
# import tensorflow as tf
# from tensorflow.contrib.layers.python.layers import initializers
#
# lstm_cell_1=tf.nn.rnn_cell.LSTMCell(
#     num_units=128,
#     use_peepholes=True,
#     initializer=initializers.xavier_initializer(),
#     num_proj=64,
#     name="LSTM_CELL_1"
# )
#
# lstm_cell_2=tf.nn.rnn_cell.LSTMCell(
#     num_units=128,
#     use_peepholes=True,
#     initializer=initializers.xavier_initializer(),
#     num_proj=64,
#     name="LSTM_CELL_2"
# )
#
#
# lstm_cell_3=tf.nn.rnn_cell.LSTMCell(
#     num_units=128,
#     use_peepholes=True,
#     initializer=initializers.xavier_initializer(),
#     num_proj=64,
#     name="LSTM_CELL_3"
# )
#
# multi_cell=tf.nn.rnn_cell.MultiRNNCell(cells=[lstm_cell_1,lstm_cell_2,lstm_cell_3])
#
# X=tf.ones(shape=(20,30,128),dtype=tf.float32)
#
# outputs,states=tf.nn.dynamic_rnn(cell=multi_cell,inputs=X,dtype=tf.float32)
#
# print("outputs:",outputs)
# print("states:",states)
#
# # print("output_size:",multi_cell.output_size)
# # print("state_size:",type(multi_cell.state_size))
# # print("state_size:",multi_cell.state_size)
# #
# # #需要先索引到具体的那层cell，然后取出具体的state状态
# # print(multi_cell.state_size[0].h)
# # print(multi_cell.state_size[0].c)
#



import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

def lstm1_test():
    #模拟一个batch为20，时间步为40，embedding大小为200的输入
    input=tf.ones(shape=(20,40,200),dtype=tf.float32,name="input")

    #lstm1,默认输出为最后一个时间步的结果
    lstm1=layers.LSTM(units=100)
    result1=lstm1(inputs=input,mask=None,training=True)
    print("result1:\n",result1)
    print("\n\n")

    #lstm2：输出为整个时间步的结果
    lstm2=layers.LSTM(units=100,return_sequences=True)
    result2 = lstm2(inputs=input, mask=None, training=True)
    print("result2:\n", result2)
    print("\n\n")

    #lstm3:输入整个时间步和状态的结果(tensorflow 1.x默认方式)
    lstm3 = layers.LSTM(units=100, return_sequences=True,return_state=True)
    result3 = lstm3(inputs=input, mask=None, training=True)
    print("result3:\n", result3)
    print("len of result3:\n",len(result3))
    print("\n\n")


def lstm2_test():
    # 模拟一个batch为20，时间步为40，embedding大小为200的输入
    input = tf.ones(shape=(20, 40, 200), dtype=tf.float32, name="input")

    # lstm1,默认输出为最后一个时间步的结果
    lstm1 = layers.LSTM(units=100)
    bilstm1=layers.Bidirectional(layer=lstm1,merge_mode="concat")
    result1 = bilstm1(inputs=input, mask=None, training=True)
    print("result1:\n", result1)
    print("\n\n")

    # # lstm2：输出为整个时间步的结果
    # lstm2 = layers.LSTM(units=100, return_sequences=True)
    # result2 = lstm2(inputs=input, mask=None, training=True)
    # print("result2:\n", result2)
    # print("\n\n")
    #
    # # lstm3:输入整个时间步和状态的结果(tensorflow 1.x默认方式)
    # lstm3 = layers.LSTM(units=100, return_sequences=True, return_state=True)
    # result3 = lstm3(inputs=input, mask=None, training=True)
    # print("result3:\n", result3)
    # print("len of result3:\n", len(result3))
    # print("\n\n")




if __name__=="__main__":
    lstm1_test()
    lstm2_test()



