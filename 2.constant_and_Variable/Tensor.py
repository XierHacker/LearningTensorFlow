from __future__ import print_function,division
import tensorflow as tf

#basic
def basic_test():
    print("基础测试")
    a=tf.constant([[1,2],[3,4]])
    b=tf.constant([[1,1],[0,1]])
    print("a:\n",a)
    print("b:\n",b)
    print("type of a:\n",type(a))
    c=tf.matmul(a,b)
    print("c:\n",c)
    print("c.numpy:\n",c.numpy())
    print("type of c.numpy():\n",type(c.numpy()))
    print("\n")

    # attribute
    print("device:", c.device)
    print("dtype:", c.dtype)
    print("shape:", type(c.shape))

    # member function

#索引测试
def index_test():
    print("索引测试")
    a = tf.constant([[1, 2], [3, 4]])
    print("a:\n", a)
    print ("a[1,:]\n",a[1,:])
    print("a[1,1]:\n",a[1,1])




def assign_value_test():
    print("赋值测试")
    a = tf.constant([[1, 2], [3, 4]])
    print("a:\n", a)
    # a[1, 1]=999 错误的赋值操作
    np_value=a.numpy()
    np_value[1,1]=999
    print("np_value:\n",np_value)



if __name__=="__main__":
    basic_test()
    index_test()
    assign_value_test()
