import tensorflow as tf
import numpy as np

#graph
graph=tf.Graph()
with graph.as_default():
    a=tf.Variable(initial_value=[[1,2],[3,4]],dtype=tf.float32,name="a")
    print(a)
    b = tf.Variable(initial_value=[[1, 1], [1, 1]], dtype=tf.float32, name="b")
    c=a+b
    cons=tf.constant(value=[1,2,3,4,5],name="cons")
    init_op=tf.global_variables_initializer()
    #Saver class
    saver=tf.train.Saver()

with tf.Session(graph=graph) as sess:
    sess.run(init_op)
    print("c:\n",sess.run(c))
    print("cons:\n",sess.run(cons))
    #save model
    path=saver.save(sess=sess,save_path="./model.ckpt")
    print("path:",path)
