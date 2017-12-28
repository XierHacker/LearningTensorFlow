import tensorflow as tf
import numpy as np

value=np.array([[0,1,2,3,4],[4,0,1,2,3],[3,4,0,1,1],[1,2,4,3,1]])
graph=tf.Graph()
session=tf.Session(graph=graph)

with graph.as_default():
    y_pred=tf.Variable(initial_value=value,name="y_pred")
    print(y_pred.shape)

    y_arg=tf.argmax(y_pred, 1)
    print("y_arg.shape:",y_arg.shape)

    #flatten
    y_pred2=tf.reshape(y_pred, [-1])
    print("y_pred2.shape:",y_pred2.shape)