import tensorflow as tf

#基本测试
def basic_test():
    print("----------------基本测试------------------")
    a = tf.Variable(initial_value=[[1, 2, 3, 4], [5, 6, 7, 8]])
    print("a:\n", a)
    print("shape of a:", a.shape)
    print("type of a.shape:", type(a.shape))
    print("a.shape[0]:", a.shape[0])
    print("type of a.shape[0]:", type(a.shape[0]))
    print("a.numpy():\n", a.numpy())
    print("type of a.numpy():", type(a.numpy()))


#索引测试
def index_test():
    print("----------------索引测试--------------------")
    a = tf.Variable(initial_value=[[1, 2, 3, 4], [5, 6, 7, 8]])
    print("a:\n", a)
    print("a[:1]:\n",a[:1])
    print("a[1,:2]：\n",a[1,:2])
    print("a[1,3]:\n",a[1,3])


#改变值测试
def assin_value_test():
    print("改变值的测试")
    a = tf.Variable(initial_value=[[1, 2, 3, 4], [5, 6, 7, 8]])
    print("a:\n", a)

    #a[1,3]=15 错误的赋值操作

    #下面的操作并不会改变a的值
    a.numpy()[1,3]=15
    print("a:\n", a)

    #通过asign来改变a的值
    np_value=a.numpy()
    np_value[1,3]=15
    print("np_vapue:",np_value)
    a.assign(value=np_value)
    print("a:\n", a)



if __name__=="__main__":
    basic_test()
    index_test()
    assin_value_test()

