from __future__ import print_function,division
import tensorflow as tf

#basic
a=tf.constant([[1,2],[3,4]])
b=tf.constant([[1,1],[0,1]])
print("a:",a)
print("b:",b)
print("type of a:",type(a))
c=tf.matmul(a,b)
print("c:",c)
print("c.numpy:\n",c.numpy())
print("type of c.numpy():",type(c.numpy()))
print("\n")


#attribute
print("device:",c.device)
print("dtype:",c.dtype)
print("shape:",type(c.shape))



#member function