import numpy as np
import tensorflow as tf

def test1():
    inputs=tf.ones(shape=(3,5,10))
    gru=tf.keras.layers.GRU(units=20,return_sequences=True,return_state=True)
    outputs,states=gru(inputs=inputs)
    print("outputs:\n",outputs)
    print("states:\n",states)


def test2():
    inputs = tf.ones(shape=(3, 5, 10))
    bi_gru=tf.keras.layers.Bidirectional(
        layer=tf.keras.layers.GRU(units=20, return_sequences=True, return_state=False)
    )
    outputs = bi_gru(inputs=inputs)
    print("outputs:\n", outputs)



def test3():
    inputs = tf.ones(shape=(3, 5, 10))
    bi_gru = tf.keras.layers.Bidirectional(
        layer=tf.keras.layers.GRU(units=20, return_sequences=True, return_state=True)
    )
    outputs,states_forward,states_backward = bi_gru(inputs=inputs)
    print("outputs:\n", outputs)
    print("states_forward:\n", states_forward)
    print("states_backward:\n", states_backward)





if __name__=="__main__":
    #test1()
    test3()