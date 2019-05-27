import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt


TIME_STEPS=28
BATCH_SIZE=128
HIDDEN_UNITS1=30
HIDDEN_UNITS=10
LEARNING_RATE=0.001
EPOCH=50

TRAIN_EXAMPLES=42000
TEST_EXAMPLES=28000

#------------------------------------Generate Data-----------------------------------------------#
#generate data
train_frame = pd.read_csv("../../Mnist/train.csv")
test_frame = pd.read_csv("../../Mnist/test.csv")

# pop the labels and one-hot coding
train_labels_frame = train_frame.pop("label")

# get values
# one-hot on labels
X_train = train_frame.astype(np.float32).values
y_train=pd.get_dummies(data=train_labels_frame).values
X_test = test_frame.astype(np.float32).values

#trans the shape to (batch,time_steps,input_size)
X_train=np.reshape(X_train,newshape=(-1,28,28))
X_test=np.reshape(X_test,newshape=(-1,28,28))
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)

#-----------------------------------------------------------------------------------------------------#
class LSTM_Model(tf.keras.Model):
    def __init__(self):
        super(LSTM_Model, self).__init__()
        self.lstm_1=layers.LSTM(units=HIDDEN_UNITS1,return_sequences=True)
        self.lstm_2=layers.LSTM(units=HIDDEN_UNITS,return_sequences=True)


    def __call__(self, inputs,training=True):
        h1=self.lstm_1(inputs=inputs,training=training)
        h2=self.lstm_2(inputs=h1,training=training)[:,-1,:]
        return h2

def train():
    model=LSTM_Model()
    optimizer=tf.keras.optimizers.Adam(LEARNING_RATE)
    cce=tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    for epoch in range(1,EPOCH+1):
        print("epoch:", epoch)
        train_losses=[]
        accus=[]
        for j in range(TRAIN_EXAMPLES//BATCH_SIZE):
            with tf.GradientTape() as tape:
                logits=model(inputs=X_train[j*BATCH_SIZE:(j+1)*BATCH_SIZE],training=True)
                #print("logits:\n",logits.shape)
                #loss
                loss=cce(y_true=y_train[j*BATCH_SIZE:(j+1)*BATCH_SIZE],y_pred=logits)
                #prediction
                correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_train[j*BATCH_SIZE:(j+1)*BATCH_SIZE], 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
                # print("loss:",loss.numpy())
                # print("accuracy:",accuracy.numpy())

                #计算梯度
                gradient = tape.gradient(target=loss, sources=model.trainable_variables)
                #print("gradient:",gradient)
                #应用梯度
                optimizer.apply_gradients(zip(gradient, model.trainable_variables))

            train_losses.append(loss)
            accus.append(accuracy)
        print("average training loss:", sum(train_losses) / len(train_losses))
        print("accuracy:",sum(accus)/len(accus))


if __name__=="__main__":
    train()