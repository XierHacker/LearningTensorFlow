import numpy as np
import tensorflow as tf

#import meta graph
saver=tf.train.import_meta_graph(meta_graph_or_file="./model.ckpt.meta")

with tf.Session() as sess:
    saver.restore(sess=sess,save_path="./model.ckpt")
    #get default graph
    graph=tf.get_default_graph()
    print(graph)

    #get tensor
    tensor_a = graph.get_tensor_by_name(name="a:0")
    print(tensor_a)
    print(sess.run(tensor_a))

