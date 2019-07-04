import os
import sys
import time
sys.path.append("../")
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
import tensorflow_datasets as tfds 
from transformer import *
from dataset import process


#load subword and restore tokenizer
tokenizer_en=tfds.features.text.SubwordTextEncoder.load_from_file(filename_prefix="../index_files/en")
tokenizer_pt=tfds.features.text.SubwordTextEncoder.load_from_file(filename_prefix="../index_files/pt")


BUFFER_SIZE = 20000
BATCH_SIZE = 64
EPOCHS=30


SOURCE_VOCAB_SIZE=tokenizer_pt.vocab_size+2
TARGET_VOCAB_SIZE=tokenizer_en.vocab_size+2
print("tokenizer_en.size:",TARGET_VOCAB_SIZE)
print("tokenizer_pt.size:",SOURCE_VOCAB_SIZE)

NUM_LAYERS=4
D_MODEL=128
DFF=512
NUM_HEADS=8
DROPOUT_RATE=0.1


CHECKPOINT_PATH="./checkpoints/train"
LOG_DIR="./log/"



def create_padding_mask(seq):
    '''
        sep should a [batch_size,max_size]
    '''
    mask=tf.cast(tf.math.equal(seq,0),tf.float32)
    # add extra dimensions so that we can add the padding to the attention logits.
    return mask[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)
    

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)

def create_masks(inp, tar):
  # Encoder padding mask
  enc_padding_mask = create_padding_mask(inp)
  
  # Used in the 2nd attention block in the decoder.
  # This padding mask is used to mask the encoder outputs.
  dec_padding_mask = create_padding_mask(inp)
  
  # Used in the 1st attention block in the decoder. 
  # It is used to pad and mask future tokens in the input received by the decoder.
  look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
  dec_target_padding_mask = create_padding_mask(tar)
  combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
  
  return enc_padding_mask, combined_mask, dec_padding_mask



class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model,warmup_steps=4000):
        super(CustomSchedule,self).__init__()
        self.d_model=tf.cast(d_model,tf.float32)
        self.warmup_steps=warmup_steps
    def __call__(self,step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def loss_fun(loss_obj,real,pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    #compute all loss
    loss_ = loss_obj(real, pred)
    mask=tf.cast(mask,dtype=loss_.dtype)
    loss_*=mask
    return tf.reduce_mean(loss_)

    
def train():
    #dataset
    train_dataset=process.get_dataset()
    train_dataset = train_dataset.cache()       # cache the dataset to memory to get a speedup while reading from it.
    train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes=([-1], [-1]))
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    #models
    transformer_obj=Transformer(NUM_LAYERS,D_MODEL,NUM_HEADS,DFF,SOURCE_VOCAB_SIZE,TARGET_VOCAB_SIZE,DROPOUT_RATE)
    learning_rate=CustomSchedule(D_MODEL)
    optimizer=tf.keras.optimizers.Adam(learning_rate,beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    loss_obj=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,reduction="none")
    
    ckpt = tf.train.Checkpoint(transformer_obj=transformer_obj,optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_PATH, max_to_keep=5)
    #log
    file_writer=tf.summary.create_file_writer(logdir=LOG_DIR)

    #if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print ('Latest checkpoint restored!!')
    

    def train_step(src_seq,tar_seq):
        tar_input_seq=tar_seq[:,:-1]
        tar_real_seq=tar_seq[:,1:]
        #get mask
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(src_seq, tar_input_seq)

        with tf.GradientTape() as tape:
            pred,_=transformer_obj(src_seq,tar_input_seq,enc_padding_mask,combined_mask,dec_padding_mask,True)
            loss=loss_fun(loss_obj,tar_real_seq,pred)
        gradients=tape.gradient(loss,transformer_obj.trainable_variables)
        optimizer.apply_gradients(zip(gradients,transformer_obj.trainable_variables))
        return loss
    
    iter_num=0
    for epoch in range(EPOCHS):
        start_time = time.time()
        for (batch, (src_seq, tar_seq)) in enumerate(train_dataset):
            loss=train_step(src_seq, tar_seq)

            #添加标量到summary
            with file_writer.as_default():
                tf.summary.scalar(name="loss",data=loss,step=iter_num)
                tf.summary.scalar(name="learning_rate",data=learning_rate(tf.constant(iter,dtype=tf.float32)),step=iter_num)
                file_writer.flush()
            print("loss:",loss.numpy())

            iter_num+=1

        ckpt_save_path = ckpt_manager.save()
        print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,ckpt_save_path))

        
    
if __name__=="__main__":
    print("start")
    train()
    # temp_learning_rate_schedule = CustomSchedule(128)
    # plt.plot(temp_learning_rate_schedule(tf.range(40000, dtype=tf.float32)))
    # plt.ylabel("Learning Rate")
    # plt.xlabel("Train Step")
    # plt.show()