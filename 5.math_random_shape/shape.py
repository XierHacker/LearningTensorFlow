import tensorflow as tf

#基本形状操作
def basic_shape():
    a=tf.constant(value=[[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])
    print("tf.shape:",tf.shape(a))
    print("tf.size:",tf.size(a))
    print("tf.rank:",tf.rank(a))

def reshape_ops():
    a = tf.constant(value=[[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])
    print("a:\n",a)
    b=tf.reshape(tensor=a,shape=(a.shape[0]*a.shape[1],-1))
    print("b:\n",b)


if __name__=="__main__":
    basic_shape()
    reshape_ops()