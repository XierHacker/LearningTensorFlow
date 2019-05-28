import pandas as pd
import numpy as np
import tensorflow as tf

EPOCH=20
BATCH_SIZE=100
TRAIN_EXAMPLES=42000
LEARNING_RATE=0.01

#------------------------Generate Data---------------------------#
#generate data
train_frame = pd.read_csv("../Mnist/train.csv")
test_frame = pd.read_csv("../Mnist/test.csv")

# pop the labels and one-hot coding
train_labels_frame = train_frame.pop("label")

# get values one-hot on labels
X_train = train_frame.astype(np.float32).values/255
y_train=pd.get_dummies(data=train_labels_frame).values
X_test = test_frame.astype(np.float32).values/255

#trans the shape to (batch,time_steps,input_size)
#X_train=np.reshape(X_train,newshape=(-1,28,28))
#X_test=np.reshape(X_test,newshape=(-1,28,28))
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)

#------------------------------------------------------------------#
class MLP(tf.keras.Model):
    def __init__(self,hidden_dim):
        super(MLP,self).__init__()
        self.linear1=tf.keras.layers.Dense(
            units=hidden_dim,
            activation=tf.nn.relu,
            use_bias=True
        )
        self.linear2=tf.keras.layers.Dense(
            units=10,
            activation=None,
            use_bias=True
        )

    def __call__(self,inputs):
        logits_1 = self.linear1(inputs)
        logits_2 = self.linear2(logits_1)
        return logits_2

def train():
    mlp=MLP(hidden_dim=200)
    optimizer=tf.keras.optimizers.SGD(LEARNING_RATE)
    checkpoint=tf.train.Checkpoint(optimizer=optimizer,model=mlp)
    for epoch in range(1,EPOCH+1):
        print("epoch:",epoch)
        train_losses = []
        accus = []
        for j in range(TRAIN_EXAMPLES//BATCH_SIZE):
            with tf.GradientTape() as tape:
                logits_2=mlp(X_train[j*BATCH_SIZE:(j+1)*BATCH_SIZE])
                entropy=tf.nn.softmax_cross_entropy_with_logits(
                    labels=y_train[j*BATCH_SIZE:(j+1)*BATCH_SIZE],
                    logits=logits_2
                )
                loss=tf.math.reduce_mean(entropy)
                #print("loss:",loss)

                #计算梯度
                gradient=tape.gradient(target=loss,sources=mlp.trainable_variables)
                #print("gradient:",gradient)
                #应用梯度
                optimizer.apply_gradients(zip(gradient,mlp.trainable_variables))

                train_losses.append(loss.numpy())
            correct_prediction = tf.equal(tf.argmax(logits_2, 1), tf.argmax(y_train[j*BATCH_SIZE:(j+1)*BATCH_SIZE], 1))
            accuracy = tf.math.reduce_mean(tf.cast(correct_prediction, "float")).numpy()
            accus.append(accuracy)
        print("average training loss:", sum(train_losses) / len(train_losses))
        print("accuracy:",sum(accus)/len(accus))

        #save checkpoint
        checkpoint.save(file_prefix="./checkpoints/test")





def test():
    mlp = MLP(hidden_dim=200)
    optimizer = tf.keras.optimizers.SGD(LEARNING_RATE)
    checkpoint = tf.train.Checkpoint(model=mlp)
    checkpoint.restore(save_path="./checkpoints/test-3")

    train_losses = []
    accus = []
    for j in range(TRAIN_EXAMPLES // BATCH_SIZE):
        logits_2 = mlp(X_train[j * BATCH_SIZE:(j + 1) * BATCH_SIZE])
        entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=y_train[j * BATCH_SIZE:(j + 1) * BATCH_SIZE],
            logits=logits_2
        )
        loss = tf.math.reduce_mean(entropy)
        train_losses.append(loss.numpy())
        correct_prediction = tf.equal(tf.argmax(logits_2, 1),
                                      tf.argmax(y_train[j * BATCH_SIZE:(j + 1) * BATCH_SIZE], 1))
        accuracy = tf.math.reduce_mean(tf.cast(correct_prediction, "float")).numpy()
        accus.append(accuracy)

    print("average training loss:", sum(train_losses) / len(train_losses))
    print("accuracy:", sum(accus) / len(accus))





if __name__=="__main__":
    istrain=False
    if istrain:
        train()
    else:
        test()