import os
import numpy as np
import tensorflow as tf

value=np.array([[1,1,1],[2,2,2],[3,3,3],[4,4,4]])
graph=tf.Graph()
with graph.as_default():
    embeddings=tf.Variable(initial_value=value,name="embeddings")
    print("embeddings.shape",embeddings.shape)

    #一维ids
    ids1=tf.Variable(initial_value=[1,2],name="ids")
    vecs1 = tf.nn.embedding_lookup(params=embeddings, ids=ids1)

    #二维ids,可以做为批处理来用
    ids2=tf.Variable(initial_value=[[1,2],[2,1],[0,2]])
    print("ids2.shape:",ids2.shape)
    vecs2=tf.nn.embedding_lookup(params=embeddings,ids=ids2)

    init_op=tf.global_variables_initializer()
    init_op = tf.global_variables_initializer()

with tf.Session(graph=graph) as sess:
    sess.run(init_op)
    embed= sess.run(embeddings)
    vec1=sess.run(vecs1)
    vec2=sess.run(vecs2)
    print("embeddings:\n",embed)
    print("vec1:\n",vec1)
    print("vec2:\n",vec2)
    print(vec2.shape)