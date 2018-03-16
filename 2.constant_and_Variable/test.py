import numpy as np
import tensorflow as tf

with tf.variable_scope("space1",reuse=False):
    v1=tf.get_variable(name="V1",shape=(2,2),dtype=tf.float32,initializer=tf.initializers.ones())
    print("name of v1:",v1.name)

    v2 = tf.get_variable(name="V2", shape=(2, 2), dtype=tf.float32, initializer=tf.initializers.zeros())
    print("name of v2:", v2.name)

with tf.variable_scope("space2",reuse=False):
    v3=tf.get_variable(name="V1",shape=(2,2),dtype=tf.float32,initializer=tf.initializers.ones())
    print("name of v3:",v3.name)

init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(v1))
    print(sess.run(v2))
    print(sess.run(v3))




