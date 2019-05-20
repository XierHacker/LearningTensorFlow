import numpy as np
import tensorflow as tf


class MLP(tf.Module):
    def __init__(self, hidden_dim):
        super(MLP, self).__init__()
        self.w_1 = tf.Variable(initial_value=tf.random.normal(shape=(784, hidden_dim)), name="w_1")
        self.b_1 = tf.Variable(initial_value=tf.zeros(shape=(hidden_dim,)), name="b_1")
        self.w_2 = tf.Variable(initial_value=tf.random.normal(shape=(hidden_dim, 10)), name="w_2")
        self.b_2 = tf.Variable(initial_value=tf.zeros(shape=(10,)), name="b_2")

    def __call__(self, inputs):
        logits_1 = tf.matmul(inputs, self.w_1) + self.b_1
        logits_1 = tf.nn.relu(logits_1)
        logits_2 = tf.matmul(logits_1, self.w_2) + self.b_2
        return logits_2


class MLP2(tf.Module):
    def __init__(self, hidden_dim):
        super(MLP2, self).__init__()
        self.linear1 = tf.keras.layers.Dense(units=hidden_dim, activation=tf.nn.relu, use_bias=True)
        self.linear2=tf.keras.layers.Dense(units=10,activation=tf.nn.relu,use_bias=True)


    def __call__(self, inputs):
        logits_1 = self.linear1(inputs)
        logits_2 = self.linear2(logits_1)
        return logits_2


class MLP3(tf.keras.Model):
    def __init__(self, hidden_dim):
        super(MLP3, self).__init__()

        self.linear1 = tf.keras.layers.Dense(units=hidden_dim, activation=tf.nn.relu, use_bias=True)
        self.linear2=tf.keras.layers.Dense(units=10,activation=tf.nn.relu,use_bias=True)


    def __call__(self, inputs):
        logits_1 = self.linear1(inputs)
        logits_2 = self.linear2(logits_1)
        return logits_2






# class MLP2(tf.keras.Model):
#     def __init__(self,hidden_dim):
#         super(MLP2,self).__init__()
#         self.linear1=tf.keras.layers.Dense(units=hidden_dim,activation=tf.nn.relu,use_bias=True)
#         self.linear2=tf.keras.layers.Dense(units=10,activation=tf.nn.relu,use_bias=True)
#         # self.model=tf.keras.Sequential()
#         # self.model.add(tf.keras.layers.Dense(units=hidden_dim,activation=tf.nn.relu,use_bias=True))
#         # self.model.add(tf.keras.layers.Dense(units=10,activation=None,use_bias=True))
#
#     def __call__(self,inputs):
#         # logits_1 = self.linear1(inputs)
#         # logits_2 = self.linear2(logits_1)
#         # return logits_2
#         pass
#
# def linear(inputs,hidden_dim,name_scope):
#     with tf.name_scope("name_scope"):
#         w_1 = tf.Variable(initial_value=tf.random.normal(shape=(784, hidden_dim)), name="w_1")
#         b_1 = tf.Variable(initial_value=tf.zeros(shape=(hidden_dim,)), name="b_1")
#         logits=tf.matmul(inputs, w_1) + b_1
#         return logits












def basic_test():
    #mlp=MLP(hidden_dim=200)
    #print("trainable_Variables:",mlp.trainable_variables)
    #print("variabels:",mlp.variables)

    # mlp2 = MLP2(hidden_dim=200)
    # result=mlp2(inputs=tf.random.normal(shape=(20,784)))
    # print("trainable_Variables:", mlp2.linear1.variables)
    # print(mlp2.linear2.variables)
    # print(mlp2.submodules)
    #print("variabels:", mlp2.variables)
    # print("type of mlp2:",type(mlp2))

    mlp3 = MLP3(hidden_dim=200)
    result = mlp3(inputs=tf.random.normal(shape=(20, 784)))
    print(mlp3.variables)
    #print("trainable_Variables:", mlp3.linear1.variables)
    #print(mlp3.linear2.variables)
    #print(mlp3.submodules)










if __name__=="__main__":
    basic_test()