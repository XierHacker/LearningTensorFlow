import numpy as np
import tensorflow as tf

class Encoder(tf.keras.Model):
    def __init__(self,vocab_size,embeddings_dim,units,batch_size):
        super(Encoder,self).__init__()
        self.word_embeddings = tf.Variable(
            initial_value=tf.random.truncated_normal(shape=(vocab_size, embeddings_dim)),
            trainable=True
        )
        self.gru=tf.keras.layers.GRU(
            units=units,
            return_sequences=True,
            return_state=True
        )


    def __call__(self,word_ids,mask,training=True):
        inputs=tf.nn.embedding_lookup(params=self.word_embeddings,ids=word_ids)
        outputs,states=self.gru(inputs=inputs,mask=mask,training=training)
        # print("encoder outputs:\n",outputs)
        # print("encoder state:\n",states)
        return outputs,states



class Decoder(tf.keras.Model):
    def __init__(self,vocab_size,embeddings_dim,units,batch_size):
        super(Decoder,self).__init__()
        self.word_embeddings = tf.Variable(
            initial_value=tf.random.truncated_normal(shape=(vocab_size, embeddings_dim)),
            trainable=True
        )
        self.gru = tf.keras.layers.GRU(
            units=units,
            return_sequences=True,
            return_state=True
        )
        self.attention = BahdanauAttention(hidden_units=units)
        self.linear=tf.keras.layers.Dense(units=vocab_size)


    def __call__(self, word_ids,pre_states,encoder_outputs):
        '''
        解码器解码一个时间步的信息
        :param word_ids: 输入的word的id，形状为[batch_size,1],因为只有一个时间步
        :param pre_states: 解码器的上一个状态,形状为[batch_size,dim]
        :param encoder_outputs: encoder的全部输出,形状为[batch_size,max_time,dim]
        :return:
        '''
        inputs=tf.nn.embedding_lookup(params=self.word_embeddings,ids=word_ids) #[batch_size,1,embedding_dim]
        #print("decoder inputs.shape",inputs)

        attention_weights, context_vector=self.attention(
            query=pre_states,
            keys=encoder_outputs,
            values=encoder_outputs
        )
        #print("attention_weights:\n",attention_weights)
        #print("contect vector:\n",context_vector)

        inputs=tf.concat(
            values=[inputs,tf.expand_dims(input=context_vector,axis=1)],
            axis=-1
        )   #[batch_size,1,embeddings_dim+encoder_dim]
        #print("decoder inputs.shape",inputs)

        #output:[batch_size,1,hidden_units] states:[batch_size,hidden_units]
        outputs,states=self.gru(inputs=inputs)
        outputs=tf.reshape(tensor=outputs,shape=(-1,outputs.shape[2])) #[batch_size,hidden_units]
        outputs=self.linear(outputs)        #[batch_size,vocab_size]

        # print("decoder outputs:\n", outputs)
        # print("decoder state:\n", states)

        return outputs,states,attention_weights



class BahdanauAttention(tf.keras.Model):
    def __init__(self,hidden_units):
        '''
        BahdanuAttention初始化函数
        :param hidden_units: attention里面的全连接的隐藏层节点数
        '''
        super(BahdanauAttention,self).__init__()
        self.dense_w1=tf.keras.layers.Dense(units=hidden_units)
        self.dense_w2=tf.keras.layers.Dense(units=hidden_units)
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
        logits_query=self.dense_w1(query)              #[batch_size,1,hidden_units]

        logits_keys=self.dense_w2(keys)                #[batch_size,max_time,hidden_units]

        logits=tf.nn.tanh(x=logits_query+logits_keys)  #[batch_size,max_time,hidden_units]

        logits_v=self.dense_v(logits)                  #[batch_size,max_time,1]

        attention_weights=tf.nn.softmax(logits=logits_v,axis=1)     #[batch_size,max_time,1]
        context_vector=tf.reduce_sum(input_tensor=attention_weights*values,axis=1)  #[batch_size,values_dim]
        #print("context_vector:", context_vector)
        return attention_weights,context_vector



if __name__=="__main__":
    # query=tf.ones(shape=(20,100),dtype=tf.float32,name="query")
    # print("query:",query)
    #
    # keys=tf.ones(shape=(20,50,200),dtype=tf.float32,name="keys")
    # print("keys:",keys)
    #
    # attention_obj=BahdanauAttention(hidden_units=100)
    # attention_obj(query=query,keys=keys,values=keys)


    src_word_ids=np.array([[9,26,7,40],[7,24,6,100],[5,4,200,300],[5,4,200,300]])
    target_word_ids = np.array([[500, 26, 7, 40,10], [7, 24, 6, 100,0], [5, 4, 200, 300,80], [5, 4, 200, 300,90]])

    encoder=Encoder(vocab_size=5230,embeddings_dim=200,units=128,batch_size=30)
    en_outputs,en_states=encoder(word_ids=src_word_ids,mask=None,training=True)


    decoder=Decoder(vocab_size=62054,embeddings_dim=200,units=128,batch_size=30)
    word_ids_one_step=target_word_ids[:,0]
    print("word_ids_one_step:\n",word_ids_one_step)
    word_ids_one_step=np.expand_dims(a=word_ids_one_step,axis=-1)
    print("word_ids_one_step:\n",word_ids_one_step)


    decoder(word_ids=word_ids_one_step,pre_states=en_states,encoder_outputs=en_outputs)