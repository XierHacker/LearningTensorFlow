import numpy as np
import tensorflow as tf

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(
            self.enc_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state = hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


class BahdanauAttention(tf.keras.Model):
    def __init__(self,attention_dim):
        '''
        BahdanuAttention初始化函数
        :param hidden_units: attention里面的全连接的隐藏层节点数
        '''
        super(BahdanauAttention,self).__init__()
        self.dense_w1=tf.keras.layers.Dense(units=attention_dim)
        self.dense_w2=tf.keras.layers.Dense(units=attention_dim)
        self.dense_v=tf.keras.layers.Dense(units=1)

    def __call__(self,query,keys,values):
        '''
        进行Bahdanau Attention操作
        :param query:  一个时刻t的query,我们一般使用解码器的某个时刻的hidden state，形状为[batch_size,query_dim]
        :param keys:   和query比较的keys, 一般使用编码器的全部输出，形状为[batch_size,max_time,keys_dim]
        :param values: 需要被attention的values,一般和key相同，你也可以使用自定的values，形状为[batch_size,max_time,values_dim]
        :return: 返回一个形状为[batch_size,max_time,1]的attention权重矩阵和形状为[batch_size,values_dim]的注意力向量
        '''
        query=tf.expand_dims(input=query,axis=1)       #[batch_size,1,query_dim]
        logits_query=self.dense_w1(query)              #[batch_size,1,attention_dim]
        logits_keys=self.dense_w2(keys)                #[batch_size,max_time,attention_dim]
        logits=tf.nn.tanh(x=logits_query+logits_keys)  #broad casting,[batch_size,max_time,hidden_units]
        logits_v=self.dense_v(logits)                  #[batch_size,max_time,1]
        attention_weights=tf.nn.softmax(logits=logits_v,axis=1)     #[batch_size,max_time,1]
        context_vector=tf.reduce_sum(input_tensor=attention_weights*values,axis=1)  #[batch_size,values_dim]
        #print("context_vector:", context_vector)
        return attention_weights,context_vector


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attention
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        # enc_output shape == (batch_size, max_length, hidden_size)
        attention_weights,context_vector= self.attention(hidden, enc_output,enc_output)
        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)
        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        # passing the concatenated vector to the GRU
        output, state = self.gru(x)
        # print("outpouts:\n",output)
        # print("state:\n",state)
        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))
        # output shape == (batch_size, vocab)
        x = self.fc(output)
        return x, state, attention_weights





if __name__=="__main__":
    # src_word_ids=np.array([[9,26,7,40],[7,24,6,100],[5,4,200,300],[5,4,200,300]])

    encoder=Encoder(vocab_size=25216,embedding_dim=256,enc_units=1024,batch_sz=64)
    sample_hidden = encoder.initialize_hidden_state()
    example_input_batch=tf.ones(shape=(64,88),dtype=tf.int32)
    sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
    print('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
    print('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))

    attention_layer = BahdanauAttention(128)
    attention_weights,attention_result= attention_layer(sample_hidden, sample_output,sample_output)
    print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
    print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))

    decoder = Decoder(13053, 256, 1024, 64)

    sample_decoder_output, _, _ = decoder(tf.random.uniform((64, 1)),sample_hidden, sample_output)

    print('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))



    # encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)

    # target_word_ids = np.array([[500, 26, 7, 40,10], [7, 24, 6, 100,0], [5, 4, 200, 300,80], [5, 4, 200, 300,90]])

    # encoder=Encoder(vocab_size=5230,embeddings_dim=200,units=128,batch_size=30)
    # en_outputs,en_states=encoder(word_ids=src_word_ids,mask=None,training=True)


    # decoder=Decoder(vocab_size=62054,embeddings_dim=200,units=128,batch_size=30)
    # word_ids_one_step=target_word_ids[:,0]
    # print("word_ids_one_step:\n",word_ids_one_step)
    # word_ids_one_step=np.expand_dims(a=word_ids_one_step,axis=-1)
    # print("word_ids_one_step:\n",word_ids_one_step)
    #
    #
    # decoder(word_ids=word_ids_one_step,pre_states=en_states,encoder_outputs=en_outputs)