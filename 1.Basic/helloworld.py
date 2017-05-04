#import tensorflow
from __future__ import print_function,division
import tensorflow as tf

#define the graph
info_op=tf.constant("hello,world")
a=tf.constant(10)
b=tf.constant(20)
add_op=tf.add(a,b)

#run graph in session
with tf.Session() as session:
    print(session.run(info_op))
    print(session.run(add_op))