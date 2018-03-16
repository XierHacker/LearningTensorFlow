import numpy as np
import tensorflow as tf

graph=tf.Graph()
np_a=np.zeros(shape=(10,1))
np_b=np.ones(shape=(10,1))
np_c=np.full(shape=(10,1),fill_value=5)
np_d=np.full(shape=(10,2,3),fill_value=1,dtype=np.float64)
with graph.as_default():
    e=[]
    a=tf.Variable(initial_value=(np_a))
    e.append(a)
    b = tf.Variable(initial_value=(np_b))
    e.append(b)
    print("e:",e)
    d = tf.Variable(initial_value=(np_d))
    e=tf.concat(values=e,axis=1)
    print("e:",e)
    print("e.shape",e.shape)

    #softmax
    softed=tf.nn.softmax(logits=e,axis=1)
    print("softed.shape",softed.shape)
    print("softed[0].shape:",softed[0].shape)

    result=[]
    for i in range(10):
        matrix=tf.matmul(tf.reshape(tensor=softed[i],shape=(1,-1)),d[i])
        result.append(matrix)

    print("result:",result)
    result=tf.concat(values=result,axis=0)
    print("result:",result)

    init=tf.global_variables_initializer()


with tf.Session(graph=graph) as sess:
    sess.run(init)
    print("a:\n",sess.run(a))
    print("b:\n", sess.run(b))
    print("b:\n", sess.run(d))
    print("e:\n", sess.run(e))
    print("softed:\n",sess.run(softed))
    print("result:\n",sess.run(result))


