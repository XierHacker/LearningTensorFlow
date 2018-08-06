import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
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
    lstm_forward=rnn.BasicLSTMCell(num_units=HIDDEN_UNITS)
    lstm_backward=rnn.BasicLSTMCell(num_units=HIDDEN_UNITS)

    outputs,(states_fw,states_bw)=tf.nn.bidirectional_dynamic_rnn(
        cell_fw=lstm_forward,
        cell_bw=lstm_backward,
        inputs=X_p,
        dtype=tf.float32
    )
    states_concat=tf.concat(values=[states_fw,states_bw],axis=1)
    print(type(states_concat))
    '''
    print("state_concat.shape:",states_concat.shape)
    finale_state=(states_concat[0],states_concat[1])
    print("type of final_state:",type(finale_state))
    c_forward=states[0].c
    print(c_forward.shape)
    c_backward=states[1].c
    print(c_backward.shape)
    c_concat=tf.concat(values=[c_forward,c_backward],axis=-1)
    print(c_concat.shape)
    '''


    #print(outputs[0].shape)
    #print(states[0].h)
    #state_h_fw=states[0].h
    #print(state_h_fw.shape)
    #outputs_fw=outputs[0]
    #outputs_bw = outputs[1]
    #output_h_fw = outputs_fw[:,-1,:]
    #print(output_h_fw.shape)
    #h=outputs_fw[:,-1,:]+outputs_bw[:,-1,:]
    #print(h.shape)
    #---------------------------------------;-----------------------------------------------------#

    #---------------------------------define loss and optimizer----------------------------------#
    cross_loss=tf.losses.softmax_cross_entropy(onehot_labels=y_p,logits=h)
    #print(loss.shape)

    correct_prediction = tf.equal(tf.argmax(h, 1), tf.argmax(y_p, 1))
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

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
            _,train_loss,accu,s_fw,o_fw=sess.run(
                    fetches=(optimizer,cross_loss,accuracy,state_h_fw,output_h_fw),
                    feed_dict={
                            X_p:X_train[j*BATCH_SIZE:(j+1)*BATCH_SIZE],
                            y_p:y_train[j*BATCH_SIZE:(j+1)*BATCH_SIZE]
                        }
            )
            train_losses.append(train_loss)
            accus.append(accu)
            #print("s_fw:",s_fw[0])
            #print("o_fw:",o_fw[0])
        print("average training loss:", sum(train_losses) / len(train_losses))
        print("accuracy:",sum(accus)/len(accus))

