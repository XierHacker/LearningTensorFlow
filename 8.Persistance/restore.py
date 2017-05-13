import tensorflow as tf
import numpy as np

#graph
graph=tf.Graph()
with graph.as_default():
    v1=tf.Variable(initial_value=tf.ones(shape=(2,2)),dtype=tf.float32,name="a")
    v2 = tf.Variable(initial_value=tf.ones(shape=(2,2)), dtype=tf.float32, name="b")
    v=v1+v2
    cons=tf.constant(value=[2,3,4,5],name="cons")
    init_op=tf.global_variables_initializer()
    saver=tf.train.Saver()

with tf.Session(graph=graph) as sess:
    #sess.run(init_op)
    #restore model
    saver.restore(sess=sess,save_path="./model.ckpt")
    print("c:\n",sess.run(v))
    print("cons:\n",sess.run(cons))