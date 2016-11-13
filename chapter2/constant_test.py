from __future__ import print_function,division
import tensorflow as tf

#build graph
a=tf.constant(1.,name="a")
print("a:",a)
print("name of a:",a.name)
b=tf.constant(1.,shape=[2,2],name="b")
print("b:",b)
print("type of b:",type(b))

#construct session
sess=tf.Session()

#run in session
result_a=sess.run(a)
print("result_a:",result_a)
print("type of result_a:",type(result_a))