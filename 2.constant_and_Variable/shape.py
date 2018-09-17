import tensorflow as tf

a=tf.Variable(initial_value=[[1,2],[3,4]])


print("a:",a)
print("a.shape:",type(a.shape[0].value))
print("a.get_shape:",a.get_shape()[0].value)