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


BUFFER_SIZE = 40000
BATCH_SIZE = 64
EPOCHS=30


SOURCE_VOCAB_SIZE=tokenizer_pt.vocab_size+2
TARGET_VOCAB_SIZE=tokenizer_en.vocab_size+2
print("tokenizer_en.size:",TARGET_VOCAB_SIZE)
print("tokenizer_pt.size:",SOURCE_VOCAB_SIZE)

NUM_LAYERS=5
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
                tf.summary.scalar(name="learning_rate",data=learning_rate(tf.constant(iter_num,dtype=tf.float32)),step=iter_num)
                file_writer.flush()
            print("loss:",loss.numpy())

            iter_num+=1

        ckpt_save_path = ckpt_manager.save()
        print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,ckpt_save_path))


def evaluate(src_sentence,tansformer_obj):
    start_token = [tokenizer_pt.vocab_size]
    end_token = [tokenizer_pt.vocab_size + 1]

    # inp sentence is portuguese, hence adding the start and end token
    src_sentence = start_token + tokenizer_pt.encode(src_sentence) + end_token
    print("src_sentence:",src_sentence)
    encoder_input = tf.expand_dims(src_sentence, 0)
    print("encoder_inputs:\n",encoder_input)
    

    # as the target is english, the first word to the transformer should be the english start token.
    decoder_input = [tokenizer_en.vocab_size]
    print("decoder_input:\n",decoder_input)
    output = tf.expand_dims(decoder_input, 0)
    print("output:\n",output)

    for i in range(40):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)
        # print("enc_padding_mask:\n",enc_padding_mask)
        # print("combine_mask:\n",combined_mask)
        # print("dec_padding_mask:\n",dec_padding_mask)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions,attention_weights=tansformer_obj(encoder_input,output,enc_padding_mask,combined_mask,dec_padding_mask,False)

        # select the last word from the seq_len dimension
        predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
    
        # return the result if the predicted_id is equal to the end token
        if tf.equal(predicted_id, tokenizer_en.vocab_size+1):
            return tf.squeeze(output, axis=0), attention_weights
        
        # concatentate the predicted_id to the output which is given to the decoder as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), attention_weights


def plot_attention_weights(attention, sentence, result, layer):
  fig = plt.figure(figsize=(16, 8))
  
  sentence = tokenizer_pt.encode(sentence)
  
  attention = tf.squeeze(attention[layer], axis=0)
  
  for head in range(attention.shape[0]):
    ax = fig.add_subplot(2, 4, head+1)
    
    # plot the attention weights
    ax.matshow(attention[head][:-1, :], cmap='viridis')

    fontdict = {'fontsize': 10}
    
    ax.set_xticks(range(len(sentence)+2))
    ax.set_yticks(range(len(result)))
    
    ax.set_ylim(len(result)-1.5, -0.5)
        
    ax.set_xticklabels(
        ['<start>']+[tokenizer_pt.decode([i]) for i in sentence]+['<end>'], 
        fontdict=fontdict, rotation=90)
    
    ax.set_yticklabels([tokenizer_en.decode([i]) for i in result 
                        if i < tokenizer_en.vocab_size], 
                       fontdict=fontdict)
    
    ax.set_xlabel('Head {}'.format(head+1))
  
  plt.tight_layout()
  plt.show()





def translate(src_sentence):
    #restore transformer from checkpoints
    #models
    transformer_obj=Transformer(NUM_LAYERS,D_MODEL,NUM_HEADS,DFF,SOURCE_VOCAB_SIZE,TARGET_VOCAB_SIZE,DROPOUT_RATE)
    #optimizer=tf.keras.optimizers.Adam(learning_rate,beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    ckpt = tf.train.Checkpoint(transformer_obj=transformer_obj)
    ckpt.restore(save_path="./checkpoints/train/ckpt-28")


    result, attention_weights = evaluate(src_sentence,transformer_obj)

    predicted_sentence = tokenizer_en.decode([i for i in result if i < tokenizer_en.vocab_size])  

    print('Input: {}'.format(src_sentence))
    print('Predicted translation: {}'.format(predicted_sentence))

    #draw_multihead_attention(attention_weights=attention_weights,src_seq=src_sentence,tar_seq=predicted_sentence,layer='decoder_layer4_block2')
    plot_attention_weights(attention_weights,src_sentence,result,'decoder_layer4_block2')


    

    


   

   
    


        
    
if __name__=="__main__":
    print("start")
    train()
    # translate("este é um problema que temos que resolver.")
    # temp_learning_rate_schedule = CustomSchedule(128)
    # plt.plot(temp_learning_rate_schedule(tf.range(40000, dtype=tf.float32)))
    # plt.ylabel("Learning Rate")
    # plt.xlabel("Train Step")
    # plt.show()
    # evaluate(src_sentence="este é um problema que temos que resolver.")