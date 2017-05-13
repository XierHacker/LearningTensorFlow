import numpy as np
import tensorflow as tf

x=np.array([[[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],
            [[0,0,0],[0,1,2],[1,1,0],[1,1,2],[2,2,0],[2,0,2],[0,0,0]],
            [[0,0,0],[0,0,0],[1,2,0],[1,1,1],[0,1,2],[0,2,1],[0,0,0]],
            [[0,0,0],[1,1,1],[1,2,0],[0,0,2],[1,0,2],[0,2,1],[0,0,0]],
            [[0,0,0],[1,0,2],[0,2,0],[1,1,2],[1,2,0],[1,1,0],[0,0,0]],
            [[0,0,0],[0,2,0],[2,0,0],[0,1,1],[1,2,1],[0,0,2],[0,0,0]],
            [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]])

#print("x[:,:,0]:\n",x[:,:,0])
#print("x[:,:,1]:\n",x[:,:,1])
#print("x[:,:,2]:\n",x[:,:,2])

W=np.array([[[[1,-1,0],[1,0,1],[-1,-1,0]],
             [[-1,0,1],[0,0,0],[1,-1,1]],
             [[-1,1,0],[-1,-1,-1],[0,0,1]]],

            [[[-1,1,-1],[-1,-1,0],[0,0,1]],
             [[-1,-1,1],[1,0,0],[0,-1,1]],
             [[-1,-1,0],[1,0,-1],[0,0,0]]]])

'''
#reshap to 4-D
#print(x.shape)

#print(x.shape)
#print(x[0][:,:,0])
'''

#this
x=np.reshape(a=x,newshape=(1,7,7,3))

#w

'''
print(W.shape)
print(W[0][:,:,0])
print(W[0][:,:,1])
print(W[0][:,:,2])
print(W[1][:,:,0])
print(W[1][:,:,1])
print(W[1][:,:,2])
'''

#this can make the W'th format match the fuction's
W=np.transpose(W,axes=(1,2,3,0))
print(W.shape)

#
b=np.array([1,0])
#define graph
graph=tf.Graph()
with graph.as_default():
   input=tf.constant(value=x,dtype=tf.float32,name="input")
   filter=tf.constant(value=W,dtype=tf.float32,name="filter")
   bias=tf.constant(value=b,dtype=tf.float32,name="bias")
   result=tf.nn.conv2d(input=input,filter=filter,strides=[1,2,2,1],padding="VALID",name="conv")+bias

with tf.Session(graph=graph) as sess:
    r=sess.run(result)
    print(r.shape)
    print(r[0][:,:,0])
    print(r[0][:,:,1])