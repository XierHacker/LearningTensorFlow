import tensorflow as tf

def gradient_test():
    #-------------------一元梯度案例---------------------------
    print("一元梯度")
    x=tf.constant(value=3.0)
    with tf.GradientTape(persistent=True,watch_accessed_variables=True) as tape:
        tape.watch(x)
        y1=2*x
        y2=x*x+2
        y3=x*x+2*x
    #一阶导数
    dy1_dx=tape.gradient(target=y1,sources=x)
    dy2_dx = tape.gradient(target=y2, sources=x)
    dy3_dx = tape.gradient(target=y3, sources=x)
    print("dy1_dx:",dy1_dx)
    print("dy2_dx:", dy2_dx)
    print("dy3_dx:", dy3_dx)


    # # -------------------二元梯度案例---------------------------
    print("二元梯度")
    x = tf.constant(value=3.0)
    y = tf.constant(value=2.0)
    with tf.GradientTape(persistent=True,watch_accessed_variables=True) as tape:
        tape.watch([x,y])
        z1=x*x*y+x*y
    # 一阶导数
    dz1_dx=tape.gradient(target=z1,sources=x)
    dz1_dy = tape.gradient(target=z1, sources=y)
    dz1_d=tape.gradient(target=z1,sources=[x,y])
    print("dz1_dx:", dz1_dx)
    print("dz1_dy:", dz1_dy)
    print("dz1_d:",dz1_d)
    print("type of dz1_d:",type(dz1_d))


def keras_test():
    input=tf.ones(shape=(20,748))
    a = tf.keras.layers.Dense(32)
    b = tf.keras.layers.Dense(32)
    print("a.variabels:",a.variables)

    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(a.variables)
        result=a(input)
        print("a.variabels:", a.variables)
        gradient=tape.gradient(result, a.variables)  # The result of this computation will be
        print("gradient:",gradient)
        # a list of `None`s since a's variables
        # are not being watched.

if __name__=="__main__":
    #gradient_test()
    keras_test()