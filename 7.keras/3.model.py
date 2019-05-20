import numpy as np
import tensorflow as tf

def functional_api_test():
    input_x=tf.keras.Input(shape=(784,))
    logits_1=tf.keras.layers.Dense(units=200,activation=tf.nn.relu,use_bias=True)(input_x)
    logits_2=tf.keras.layers.Dense(units=100,activation=None,use_bias=True)(logits_1)
    pred=tf.keras.layers.Dense(units=10,activation=tf.nn.softmax,use_bias=True)(logits_2)

    model=tf.keras.Model(inputs=input_x,outputs=pred)

    #compile你想要的训练配置
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.categorical_crossentropy,
        metrics=["accuracy"]
    )
    #keras.layers.Dense()



def subclass_test():
    pass




if __name__=="__main__":
    functional_api_test()