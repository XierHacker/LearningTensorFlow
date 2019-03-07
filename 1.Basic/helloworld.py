from __future__ import print_function,division
import tensorflow as tf

#define the graph
info=tf.constant("hello,world")
a=tf.constant(10)
b=tf.constant(20)
c=tf.add(a,b)

print("info:",info)
print("type of info:",type(info))
print(info.numpy())
print("type of info.numpy()",type(info.numpy()))
print("\n\n")

print("a:",a)
print("type of a:",type(a))
print(a.numpy())
print("type of a.numpy()",type(a.numpy()))


print("b:",b)
print("c:",c)
