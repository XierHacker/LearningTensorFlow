import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import parameter


class LSTM_Model(tf.keras.Model):
    def __init__(self):
        super(LSTM_Model,self).__init__()
        self.word_embeddings = tf.Variable(
            initial_value=tf.random.truncated_normal(shape=(parameter.VOCAB_SIZE, parameter.EMBEDDINGS_DIM)),
            trainable=True
        )
        self.lstm_1=layers.Bidirectional(
            layer=layers.LSTM(units=parameter.HIDDEN_UNITS,return_sequences=True),
            merge_mode="concat"
        )
        self.lstm_2=layers.Bidirectional(
            layer=layers.LSTM(units=parameter.HIDDEN_UNITS, return_sequences=True),
            merge_mode="concat"
        )
        self.lstm_3 = layers.Bidirectional(
            layer=layers.LSTM(units=parameter.HIDDEN_UNITS, return_sequences=True),
            merge_mode="concat"
        )
        self.linear=layers.Dense(units=parameter.CLASS_NUM)


    def __call__(self,word_ids,mask,training=True):
        #embeddings look up
        inputs=tf.nn.embedding_lookup(params=self.word_embeddings,ids=word_ids)
        h1=self.lstm_1(inputs=inputs,mask=mask,training=training)
        h2=self.lstm_2(inputs=h1,mask=mask,training=training)
        h3 = self.lstm_3(inputs=h2,mask=mask,training=training)
        logits=self.linear(h3)
        return logits
