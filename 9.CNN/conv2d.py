import numpy as np
import tensorflow as tf

x=np.array([[[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],
            [[0,0,0],[0,1,2],[1,1,0],[1,1,2],[2,2,0],[2,0,2],[0,0,0]],
            [[0,0,0],[0,0,0],[1,2,0],[1,1,1],[0,1,2],[0,2,1],[0,0,0]],
            [[0,0,0],[1,1,1],[1,2,0],[0,0,2],[1,0,2],[0,2,1],[0,0,0]],
            [[0,0,0],[1,0,2],[0,2,0],[1,1,2],[1,2,0],[1,1,0],[0,0,0]],
            [[0,0,0],[0,2,0],[2,0,0],[0,1,1],[1,2,1],[0,0,2],[0,0,0]],
            [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]])


W=np.array([[[[1,-1,0],[1,0,1],[-1,-1,0]],
             [[-1,0,1],[0,0,0],[1,-1,1]],
             [[-1,1,0],[-1,-1,-1],[0,0,1]]],

            [[[-1,1,-1],[-1,-1,0],[0,0,1]],
             [[-1,-1,1],[1,0,0],[0,-1,1]],
             [[-1,-1,0],[1,0,-1],[0,0,0]]]])

print("input:\n")
print("x[:,:,0]:\n",x[:,:,0])
print("x[:,:,1]:\n",x[:,:,1])
print("x[:,:,2]:\n",x[:,:,2])

print("filter:")
print("W[0][:,:,0]:\n",W[0][:,:,0])
print("W[0][:,:,1]:\n",W[0][:,:,1])
print("W[0][:,:,2]:\n",W[0][:,:,2])
print("W[1][:,:,0]:\n",W[1][:,:,0])
print("W[1][:,:,1]:\n",W[1][:,:,1])
print("W[1][:,:,2]:\n",W[1][:,:,2])

#this
x=np.reshape(a=x,newshape=(1,7,7,3))
W=np.transpose(W,axes=(1,2,3,0))        #weights,[height,width,in_channels,out_channels]
print(W.shape)
b=np.array([1,0])                       #bias

input = tf.constant(value=x, dtype=tf.float32, name="input")
filter = tf.constant(value=W, dtype=tf.float32, name="filter")
bias = tf.constant(value=b, dtype=tf.float32, name="bias")
out=tf.nn.conv2d(input=input,filters=filter,strides=2,padding="VALID",name="conv2d")+bias

print(out[0][:,:,0])
print(out[0][:,:,1])
