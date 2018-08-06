import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt


TIME_STEPS=10
BATCH_SIZE=128
HIDDEN_UNITS=1
LEARNING_RATE=0.001
EPOCH=150

TRAIN_EXAMPLES=11000
TEST_EXAMPLES=1100

#------------------------------------Generate Data-----------------------------------------------#
#generate data
def generate(seq):
    X=[]
    y=[]
    for i in range(len(seq)-TIME_STEPS):
        X.append([seq[i:i+TIME_STEPS]])
        y.append([seq[i+TIME_STEPS]])
    return np.array(X,dtype=np.float32),np.array(y,dtype=np.float32)

#s=[i for i in range(30)]
#X,y=generate(s)
#print(X)
#print(y)

seq_train=np.sin(np.linspace(start=0,stop=100,num=TRAIN_EXAMPLES,dtype=np.float32))
seq_test=np.sin(np.linspace(start=100,stop=110,num=TEST_EXAMPLES,dtype=np.float32))

#plt.plot(np.linspace(start=0,stop=100,num=10000,dtype=np.float32),seq_train)

#plt.plot(np.linspace(start=100,stop=110,num=1000,dtype=np.float32),seq_test)
#plt.show()

X_train,y_train=generate(seq_train)
#print(X_train.shape,y_train.shape)
X_test,y_test=generate(seq_test)

#reshape to (batch,time_steps,input_size)
X_train=np.reshape(X_train,newshape=(-1,TIME_STEPS,1))
X_test=np.reshape(X_test,newshape=(-1,TIME_STEPS,1))

#draw y_test
plt.plot(range(1000),y_test[:1000,0],"r*")
#print(X_train.shape)
#print(X_test.shape)


#-----------------------------------------------------------------------------------------------------#


#--------------------------------------Define Graph---------------------------------------------------#
graph=tf.Graph()
with graph.as_default():

    #------------------------------------construct LSTM------------------------------------------#
    #place hoder
    X_p=tf.placeholder(dtype=tf.float32,shape=(None,TIME_STEPS,1),name="input_placeholder")
    y_p=tf.placeholder(dtype=tf.float32,shape=(None,1),name="pred_placeholder")

    #lstm instance
    lstm_cell=rnn.BasicLSTMCell(num_units=HIDDEN_UNITS)

    #initialize to zero
    init_state=lstm_cell.zero_state(batch_size=BATCH_SIZE,dtype=tf.float32)

    #dynamic rnn
    outputs,states=tf.nn.dynamic_rnn(cell=lstm_cell,inputs=X_p,initial_state=init_state,dtype=tf.float32)
    #print(outputs.shape)
    h=outputs[:,-1,:]
    #print(h.shape)
    #--------------------------------------------------------------------------------------------#

    #---------------------------------define loss and optimizer----------------------------------#
    mse=tf.losses.mean_squared_error(labels=y_p,predictions=h)
    #print(loss.shape)
    optimizer=tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss=mse)


    init=tf.global_variables_initializer()


#-------------------------------------------Define Session---------------------------------------#
with tf.Session(graph=graph) as sess:
    sess.run(init)
    for epoch in range(1,EPOCH+1):
        results = np.zeros(shape=(TEST_EXAMPLES, 1))
        train_losses=[]
        test_losses=[]
        print("epoch:",epoch)
        for j in range(TRAIN_EXAMPLES//BATCH_SIZE):
            _,train_loss=sess.run(
                    fetches=(optimizer,mse),
                    feed_dict={
                            X_p:X_train[j*BATCH_SIZE:(j+1)*BATCH_SIZE],
                            y_p:y_train[j*BATCH_SIZE:(j+1)*BATCH_SIZE]
                        }
            )
            train_losses.append(train_loss)
        print("average training loss:", sum(train_losses) / len(train_losses))


        for j in range(TEST_EXAMPLES//BATCH_SIZE):
            result,test_loss=sess.run(
                    fetches=(h,mse),
                    feed_dict={
                            X_p:X_test[j*BATCH_SIZE:(j+1)*BATCH_SIZE],
                            y_p:y_test[j*BATCH_SIZE:(j+1)*BATCH_SIZE]
                        }
            )
            results[j*BATCH_SIZE:(j+1)*BATCH_SIZE]=result
            test_losses.append(test_loss)
        print("average test loss:", sum(test_losses) / len(test_losses))
        plt.plot(range(1000),results[:1000,0])
    plt.show()













