import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import model


#define strategy
strategy=tf.distribute.MirroredStrategy()
print("num devices:",strategy.num_replicas_in_sync)

#parameters
BATCH_SIZE_PER_REPLICA=4
BATCH_SIZE=BATCH_SIZE_PER_REPLICA*strategy.num_replicas_in_sync
print("batch_size_per_replica:",BATCH_SIZE_PER_REPLICA)
print("batch_size:",BATCH_SIZE)

CLASS_NUM=10
EPOCHS=10

#model save path
CHECK_POINTS_PATH="./checkpoints/cnn"



print(tf.__version__)

(x_train, y_train), (x_test, y_test)=tf.keras.datasets.fashion_mnist.load_data()

#expand dim to use convlution 2D
x_train=np.expand_dims(a=x_train,axis=-1)/np.float32(255)
train_dataset=tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(BATCH_SIZE)
# for records in train_dataset:
#     print("records:\n",records[0])
#     print("records:\n",records[1])
    

x_test=np.expand_dims(a=x_test,axis=-1)/np.float32(255)
test_dataset=tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(BATCH_SIZE)

#
# plt.imshow(X=x_train[1,:,:,0])
# plt.show()


def train_step():
    pass

def train():
    #trans dataset to distribute dataset
    with strategy.scope():
        train_dist_dataset=strategy.experimental_distribute_dataset(train_dataset)
        test_dist_dataset=strategy.experimental_distribute_dataset(test_dataset)

    #define loss
    with strategy.scope():
        SCC=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.NONE)
        def compute_loss(labels,predictions):
            device_losses=SCC(labels,predictions)
            #print("device_loss")
            device_loss=tf.nn.compute_average_loss(per_example_loss=device_losses,global_batch_size=BATCH_SIZE)
            return device_loss
    
    #define metrics

    # model and optimizer must be created under `strategy.scope`.
    with strategy.scope():
        cnn=model.CNN(filters=64,kernel_size=3,strides=1,activation=tf.nn.relu)
        #model = create_model()
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=cnn)

    #basic train step in one device
    with strategy.scope():
        def train_step(inputs):
            x,y=inputs
            with tf.GradientTape() as tape:
                logits=cnn(x)
                loss=compute_loss(labels=y,predictions=logits)
            
            gradients=tape.gradient(target=loss,sources=cnn.trainable_variables)
            optimizer.apply_gradients(zip(gradients,cnn.trainable_variables))
            return loss
    
    #distribute train_step use basic train step
    with strategy.scope():
        def dist_train_step(dataset_inputs):
            replica_losses=strategy.experimental_run_v2(fn=train_step,args=(dataset_inputs,))
            #print("replica_losses:\n",replica_losses)
            return strategy.reduce(reduce_op=tf.distribute.ReduceOp.SUM,value=replica_losses,axis=None)

    for epoch in range(EPOCHS):
        epoch_loss=0.0
        num_batchs=0
        for records in train_dist_dataset:
            epoch_loss+=dist_train_step(records)
            num_batchs+=1
        epoch_loss=epoch_loss/num_batchs

        print("epoch_loss:",epoch_loss)




if __name__=="__main__":
    train()

