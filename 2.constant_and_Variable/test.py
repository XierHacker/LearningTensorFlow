import tensorflow as tf
import numpy as np

#define graph
w=tf.Variable(initial_value=[[1,2],[3,4]],dtype=tf.float32,name="w")
x=tf.Variable(initial_value=[[1,0],[0,1]],dtype=tf.float32,name="x")

#like Tensor
y=tf.matmul(w,x,name="y")

a=tf.placeholder(dtype=tf.float32,shape=(2,2),name="a")

print(y)

init_op=tf.global_variables_initializer()

#define session
sess=tf.Session()
#run init_op first
sess.run(fetches=init_op)

print(sess.run(fetches=w))
print(sess.run(fetches=x))
print(sess.run(fetches=y))




