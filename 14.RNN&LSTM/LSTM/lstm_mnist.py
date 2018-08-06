import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from tensorflow.contrib.layers.python.layers import initializers
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
#print(X_train.shape)
#print(y_dummy.shape)
#print(X_test.shape)

#-----------------------------------------------------------------------------------------------------#

#--------------------------------------Define Graph---------------------------------------------------#
graph=tf.Graph()
with graph.as_default():

    #------------------------------------construct LSTM------------------------------------------#
    #place hoder
    X_p=tf.placeholder(dtype=tf.float32,shape=(None,TIME_STEPS,28),name="input_placeholder")
    y_p=tf.placeholder(dtype=tf.float32,shape=(None,10),name="pred_placeholder")

    #lstm instance
    #lstm_cell1=rnn.BasicLSTMCell(num_units=HIDDEN_UNITS1)
    lstm_cell=tf.nn.rnn_cell.LSTMCell(
        num_units=HIDDEN_UNITS,
        use_peepholes=True,
        initializer=initializers.xavier_initializer(),
        num_proj=HIDDEN_UNITS
    )

    #multi_lstm=rnn.MultiRNNCell(cells=[lstm_cell1,lstm_cell])

    #initialize to zero
    init_state=lstm_cell.zero_state(batch_size=BATCH_SIZE,dtype=tf.float32)

    '''
    #dynamic rnn
    outputs,states=tf.nn.dynamic_rnn(cell=multi_lstm,inputs=X_p,initial_state=init_state,dtype=tf.float32)
    #print(outputs.shape)
    h=outputs[:,-1,:]
    #print(h.shape)
    '''

    outputs=[]
    state=init_state
    with tf.variable_scope("RNN"):
        for time_step in range(TIME_STEPS):
            if time_step>0:
                tf.get_variable_scope().reuse_variables()
            #shape of output is [batch_size,units_size],input:[batch_size,input_size]
            output,state=lstm_cell(inputs=X_p[:,time_step,:],state=state)
            #print(output.shape)
            outputs.append(output)
        #print(len(outputs))
    h=outputs[-1]
    #print(h.shape)
    #--------------------------------------------------------------------------------------------#

    #---------------------------------define loss and optimizer----------------------------------#
    cross_loss=tf.losses.softmax_cross_entropy(onehot_labels=y_p,logits=h)
    #print(loss.shape)

    correct_prediction = tf.equal(tf.argmax(h, 1), tf.argmax(y_p, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    optimizer=tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss=cross_loss)

    init=tf.global_variables_initializer()


#-------------------------------------------Define Session---------------------------------------#
with tf.Session(graph=graph) as sess:
    sess.run(init)
    for epoch in range(1,EPOCH+1):
        #results = np.zeros(shape=(TEST_EXAMPLES, 10))
        train_losses=[]
        accus=[]
        #test_losses=[]
        print("epoch:",epoch)
        for j in range(TRAIN_EXAMPLES//BATCH_SIZE):
            _,train_loss,accu=sess.run(
                    fetches=(optimizer,cross_loss,accuracy),
                    feed_dict={
                            X_p:X_train[j*BATCH_SIZE:(j+1)*BATCH_SIZE],
                            y_p:y_train[j*BATCH_SIZE:(j+1)*BATCH_SIZE]
                        }
            )
            train_losses.append(train_loss)
            accus.append(accu)
        print("average training loss:", sum(train_losses) / len(train_losses))
        print("accuracy:",sum(accus)/len(accus))