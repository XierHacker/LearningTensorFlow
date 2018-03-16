import numpy as np
import tensorflow as tf

batch_dim_1=np.array([[1,2,3,4],[4,5,6,7],[7,8,9,10]])
print("batch_dim:\n",batch_dim_1)

batch_dim_2=np.array([[3,4,5,6],[9,10,11,12],[13,14,15,16]])
print("batch_dim:\n",batch_dim_2)

graph=tf.Graph()
with graph.as_default():
    a=tf.Variable(initial_value=batch_dim_1)
    b=tf.Variable(initial_value=batch_dim_2)
    result=(a,b)
    print("result:",result)
    result=tf.concat(values=[a,b],axis=0)
    print(result)
    result2=tf.reshape(tensor=result,shape=(2,3,-1))
    print("result2:",result2)
    result3=tf.transpose(a=result2,perm=(1,0,2))
    print("result3:", result3)
    shape=result3.get_shape().as_list()
    print(shape)

    init=tf.global_variables_initializer()

with tf.Session(graph=graph) as sess:
    sess.run(init)
    print("result:\n",sess.run(result))
    print("result2:\n", sess.run(result2))
    print("result3:\n", sess.run(result3))