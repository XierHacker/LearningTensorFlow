import tensorflow as tf
import numpy as np

c=tf.constant(value=1)
#print(assert c.graph is tf.get_default_graph())
print(c.graph)
print(tf.get_default_graph())

g=tf.Graph()
print("g:",g)
with g.as_default():
    d=tf.constant(value=2)
    print(d.graph)
    #print(g)

g2=tf.Graph()
print("g2:",g2)
g2.as_default()
e=tf.constant(value=15)
print(e.graph)

