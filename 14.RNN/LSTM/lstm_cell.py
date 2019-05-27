'''
tf.keras.layers.LSTMCell

__init__(
    units,
    activation='tanh',
    recurrent_activation='hard_sigmoid',
    use_bias=True,
    kernel_initializer='glorot_uniform',
    recurrent_initializer='orthogonal',
    bias_initializer='zeros',
    unit_forget_bias=True,
    kernel_regularizer=None,
    recurrent_regularizer=None,
    bias_regularizer=None,
    kernel_constraint=None,
    recurrent_constraint=None,
    bias_constraint=None,
    dropout=0.0,
    recurrent_dropout=0.0,
    implementation=1,
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
- unit_forget_bias: Boolean. If True, add 1 to the bias of the forget gate at initialization. Setting it to true will also force bias_initializer="zeros". This is recommended in Jozefowicz et al.
- kernel_regularizer: Regularizer function applied to the kernel weights matrix.
- recurrent_regularizer: Regularizer function applied to the recurrent_kernel weights matrix.
- bias_regularizer: Regularizer function applied to the bias vector.
- kernel_constraint: Constraint function applied to the kernel weights matrix.
- recurrent_constraint: Constraint function applied to the recurrent_kernel weights matrix.
- bias_constraint: Constraint function applied to the bias vector.
- dropout: inputs线性变换时候的dropout比率
- recurrent_dropout: recurrent state线性变换时候的dropout比率
- implementation: Implementation mode, either 1 or 2. Mode 1 will structure its operations as a larger number of smaller dot products and additions, whereas mode 2 will batch them into fewer, larger operations. These modes will have different performance profiles on different hardware and for different applications.



调用时参数:
- inputs: A 2D tensor.
- states: List of state tensors corresponding to the previous timestep.
- training: 布尔值，表示这层是否表现为训练模式或者是前向计算模式。在dropout或者recurrent_dropout被使用的时候会有区别

'''


