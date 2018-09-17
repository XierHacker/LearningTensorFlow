import numpy as np
import pandas as pd
import tensorflow as tf
import model
import parameter


BATCH_SIZE=128
LEARNING_RATE=0.005
EPOCH=50

TRAIN_EXAMPLES=42000
TEST_EXAMPLES=28000

def train(train_set,train_lables):
    # place hoder
    X_p = tf.placeholder(dtype=tf.float32, shape=(None, 28, 28,1), name="input_placeholder")
    y_p = tf.placeholder(dtype=tf.float32, shape=(None, 10), name="pred_placeholder")

    m=model.Model()
    regularizer = tf.contrib.layers.l2_regularizer(0.0001)
    logits=m.forward(input=X_p,regularizer=regularizer)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=y_p, logits=logits)
    # print(loss.shape)
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_p, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss=loss)
    init = tf.global_variables_initializer()

    # -------------------------------------------Define Session---------------------------------------#
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(1, EPOCH + 1):
            # results = np.zeros(shape=(TEST_EXAMPLES, 10))
            train_losses = []
            accus = []
            # test_losses=[]
            print("epoch:", epoch)
            for j in range(TRAIN_EXAMPLES // BATCH_SIZE):
                _, train_loss, accu= sess.run(
                    fetches=(optimizer, loss, accuracy),
                    feed_dict={
                        X_p: X_train[j * BATCH_SIZE:(j + 1) * BATCH_SIZE],
                        y_p: y_train[j * BATCH_SIZE:(j + 1) * BATCH_SIZE]
                    }
                )
                train_losses.append(train_loss)
                accus.append(accu)
                # print("s_fw:",s_fw[0])
                # print("o_fw:",o_fw[0])
            print("train_losses.size",len(train_losses))
            print("accus.size",len(accus))
            print("average training loss:", sum(train_losses) / len(train_losses))
            print("accuracy:", sum(accus) / len(accus))





if __name__=="__main__":
    # ------------------------------------Generate Data-----------------------------------------------#
    # generate data
    train_frame = pd.read_csv("../../Mnist/train.csv")
    test_frame = pd.read_csv("../../Mnist/test.csv")

    # pop the labels and one-hot coding
    train_labels_frame = train_frame.pop("label")

    # get values
    # one-hot on labels
    X_train = train_frame.astype(np.float32).values
    y_train = pd.get_dummies(data=train_labels_frame).values
    X_test = test_frame.astype(np.float32).values

    # trans the shape to (batch,time_steps,input_size)
    X_train = np.reshape(X_train, newshape=(-1, 28, 28, 1))
    X_test = np.reshape(X_test, newshape=(-1, 28, 28, 1))
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)

    # -----------------------------------------------------------------------------------------------------#
    train(X_train,y_train)



