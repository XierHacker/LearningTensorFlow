import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import matplotlib.pyplot as plt


TIME_STEPS=10
BATCH_SIZE=32
HIDDEN_UNITS=28

#----------------------------------------------------------------------------#
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

seq_train=np.sin(np.linspace(start=0,stop=100,num=10000,dtype=np.float32))
seq_test=np.sin(np.linspace(start=100,stop=110,num=1000,dtype=np.float32))

#plt.plot(np.linspace(start=0,stop=100,num=10000,dtype=np.float32),seq_train)

#plt.plot(np.linspace(start=100,stop=110,num=1000,dtype=np.float32),seq_test)
#plt.show()

X_train,y_train=generate(seq_train)
X_test,y_test=generate(seq_test)

#-------------------------------------------------------------------------------#

graph=tf.Graph()
with graph.as_default():
    #place hoder
    X_p=
    lstm_cell=rnn.BasicLSTMCell(num_units=HIDDEN_UNITS)





