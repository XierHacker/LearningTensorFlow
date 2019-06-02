import numpy as np
import tensorflow as tf



class Encoder(tf.keras.Model):
    def __init__(self):
        super(Encoder,self).__init__()

    def __call__(self, *args, **kwargs):
        pass



class Decoder(tf.keras.Model):
    def __init__(self):
        super(Decoder,self).__init__()

    def __call__(self, *args, **kwargs):
        pass




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
        :return: 返回一个形状为[batch_size,max_time,1]的attention权重矩阵和
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
    query=tf.ones(shape=(20,100),dtype=tf.float32,name="query")
    print("query:",query)

    keys=tf.ones(shape=(20,50,200),dtype=tf.float32,name="keys")
    print("keys:",keys)

    attention_obj=BahdanauAttention(hidden_units=100)
    attention_obj(query=query,keys=keys,values=keys)




