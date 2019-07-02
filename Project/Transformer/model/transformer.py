import os
import sys
sys.path.append("../")
import numpy as np 
import tensorflow as tf 
from utils.display import draw_positional_encodings


def positional_encoding(max_pos,d_model):
    '''
        《Attention Is All You Need》 3.5
    '''
    pos=np.arange(max_pos)[:,np.newaxis]
    i=np.arange(d_model)[np.newaxis, :]
    #compute whole PE
    PE=(pos*1.0)/np.power(10000, (2 * (i//2)) / np.float32(d_model))
    # apply sin to even indices in the array; 2i
    sines = np.sin(PE[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    cosines = np.cos(PE[:, 1::2])
    # concat
    PE = np.concatenate([sines, cosines], axis=-1)
    PE = PE[np.newaxis, ...]
    return tf.cast(PE, dtype=tf.float32)



def scaled_dot_product_attention(q, k, v, mask):
    '''
        《Attention Is All You Need》 3.2.1

        q, k, v must have matching leading dimensions.
        k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
        The mask has different shapes depending on its type(padding or look ahead) 
        but it must be broadcastable for addition.
  
        Args:
            q: query shape == (..., seq_len_q, depth)
            k: key shape == (..., seq_len_k, depth)
            v: value shape == (..., seq_len_v, depth_v)
            mask: Float tensor with shape broadcastable 
                to (..., seq_len_q, seq_len_k). Defaults to None.
    
        Returns:
            output, attention_weights
    '''
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    # softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self,d_model,num_heads):
        super(MultiHeadAttention,self).__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        assert self.d_model % self.num_heads==0     #make d_model is 
        self.depth=self.d_model//self.num_heads     #depth of each head
        
        #linear transform of Q,K,V
        self.linear_q=tf.keras.layers.Dense(units=self.d_model)
        self.linear_k=tf.keras.layers.Dense(units=self.d_model)
        self.linear_v=tf.keras.layers.Dense(units=self.d_model)
        #linear transform of outputs
        self.linear=tf.keras.layers.Dense(units=self.d_model)

    def split_heads(self, x, batch_size):
        '''
            Split the last dimension into (num_heads, depth).
            Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        '''
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))     #(batch_size, seq_len,num_heads, depth)
        return tf.transpose(x, perm=[0, 2, 1, 3])               #(batch_size, num_heads, seq_len, depth)

    def __call__(self,q, k, v, mask):
        batch_size=tf.shape(q)[0]

        q=self.linear_q(q)      # (batch_size, seq_len, d_model)
        k=self.linear_k(k)      # (batch_size, seq_len, d_model)
        v=self.linear_v(v)      # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
        
        output=self.linear(concat_attention)            # (batch_size, seq_len_q, d_model)
        return output,attention_weights


class FFN(tf.keras.layers.Layer):
    '''
        Point wise feed forward network
        《attention is all you need》 3.3
    '''
    def __init__(self,dff,d_model):
        super(FFN,self).__init__()
        self.dff=dff
        self.d_model=d_model
        self.linear_1=tf.keras.layers.Dense(units=dff,activation="relu")
        self.linear_2=tf.keras.layers.Dense(units=d_model)

    def __call__(self,x):
        x=self.linear_1(x)
        x=self.linear_2(x)
        return x

class TransformerEncoderBlock(tf.keras.layers.Layer):
    '''
        basic transformer encoder blocks
        《attention is all you need》 3.1
    '''
    def __init__(self, d_model,num_heads,dff,dropout_rate=0.1):
        super(TransformerEncoderBlock,self).__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.dff=dff
        self.dropout_rate=dropout_rate
        # Multi-Head Attention
        self.multi_head_attention=MultiHeadAttention(d_model=self.d_model,num_heads=self.num_heads)
        # Feed Forward Network
        self.ffn=FFN(dff=self.dff,d_model=self.d_model)
        #layer norm
        self.layernorm_1=tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm_2=tf.keras.layers.LayerNormalization(epsilon=1e-6)
        #dropout
        self.dropout_1=tf.keras.layers.Dropout(rate=self.dropout_rate)
        self.dropout_2=tf.keras.layers.Dropout(rate=self.dropout_rate)

    def __call__(self,x,mask,training):
        #through multi-head attention
        attention_output,attention_weights=self.multi_head_attention(x,x,x,mask)    # (batch_size, input_seq_len, d_model)
        #dropout
        attention_output=self.dropout_1(attention_output,training=training)         
        #residual and layer norm
        out1=self.layernorm_1(x+attention_output)                                   # (batch_size, input_seq_len, d_model)

        #through Feed Forward Network
        ffn_output=self.ffn(out1)                   # (batch_size, input_seq_len, d_model)                                 
        #dropout
        ffn_output=self.dropout_2(ffn_output,training=training)
        #residual and layer norm
        out2=self.layernorm_2(out1+ffn_output)      # (batch_size, input_seq_len, d_model)

        return out2




class TransformerDecoderBlock(tf.keras.layers.Layer):
    '''
        basic transformer encoder blocks
        《attention is all you need》 3.1
    '''
    def __init__(self, d_model,num_heads,dff,dropout_rate=0.1):
        super(TransformerDecoderBlock,self).__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.dff=dff
        self.dropout_rate=dropout_rate

        # Multi-Head Attention
        self.multi_head_attention_1=MultiHeadAttention(d_model=self.d_model,num_heads=self.num_heads)
        self.multi_head_attention_2=MultiHeadAttention(d_model=self.d_model,num_heads=self.num_heads)

        # Feed Forward Network
        self.ffn=FFN(dff=self.dff,d_model=self.d_model)

        #layer norm
        self.layernorm_1=tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm_2=tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm_3=tf.keras.layers.LayerNormalization(epsilon=1e-6)

        #dropout
        self.dropout_1=tf.keras.layers.Dropout(rate=self.dropout_rate)
        self.dropout_2=tf.keras.layers.Dropout(rate=self.dropout_rate)
        self.dropout_3=tf.keras.layers.Dropout(rate=self.dropout_rate)

    
    def __call__(self,x,enc_output,look_ahead_mask,padding_mask,training):
        # enc_output.shape == (batch_size, input_seq_len, d_model)
        #through first multi-head attention,(batch_size, input_seq_len, d_model)
        attention_output_1,attention_weights_1=self.multi_head_attention_1(x,x,x,look_ahead_mask)
        attention_output_1=self.dropout_1(attention_output_1,training=training)
        out1=self.layernorm_1(attention_output_1+x)

        #through second multi-head attention,(batch_size, input_seq_len, d_model)
        attention_output_2,attention_weights_2=self.multi_head_attention_1(out1,enc_output,enc_output,padding_mask)
        attention_output_2=self.dropout_2(attention_output_2,training=training)
        out2=self.layernorm_2(attention_output_2+out1)

        #through Feed Forward Network
        ffn_output=self.ffn(out2)                   # (batch_size, input_seq_len, d_model)                                 
        ffn_output=self.dropout_3(ffn_output,training=training)
        out3=self.layernorm_2(out2+ffn_output)      # (batch_size, input_seq_len, d_model)

        return out3,attention_weights_1,attention_weights_2





        
            
    




if __name__=="__main__":
    # pos_encoding = positional_encoding(50, 512)
    # print (pos_encoding.shape)
    # #draw_positional_encodings(pos_encoding)

    # padding_mask(seq=[[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])

    def print_out(q, k, v):
        temp_out, temp_attn = scaled_dot_product_attention(q, k, v, None)
        print ('Attention weights are:')
        print (temp_attn)
        print ('Output is:')
        print (temp_out)

    np.set_printoptions(suppress=True)

    temp_k = tf.constant([[10,0,0],[0,10,0],[0,0,10],[0,0,10]], dtype=tf.float32)  # (4, 3)
    print("temp_k.shape:",temp_k.shape)

    temp_v = tf.constant([[ 1,0],[ 10,0],[ 100,5],[1000,6]], dtype=tf.float32)  # (4, 2)

    # This `query` aligns with the second `key`,so the second `value` is returned.
    temp_q = tf.constant([[0, 10, 0]], dtype=tf.float32)  # (1, 3)
    print_out(temp_q, temp_k, temp_v)

    # This query aligns with a repeated key (third and fourth), so all associated values get averaged.
    temp_q = tf.constant([[0, 0, 10]], dtype=tf.float32)  # (1, 3)
    print_out(temp_q, temp_k, temp_v)

    temp_mha = MultiHeadAttention(d_model=512, num_heads=8)
    y = tf.random.uniform((1, 60, 512))  # (batch_size, encoder_sequence, d_model)
    out, attn = temp_mha(q=y, k=y, v=y, mask=None)
    print("out.shape:",out.shape)
    print("attn.shape:",attn.shape)

    sample_ffn = FFN(2048, 512)
    print("ffn.shape:",sample_ffn(tf.random.uniform((64, 50, 512))).shape)
    

    sample_encoder_block = TransformerEncoderBlock(512, 8, 2048)
    sample_encoder_layer_output = sample_encoder_block(tf.random.uniform((64, 43, 512)), None, False)
    print("transformer block output shape:",sample_encoder_layer_output.shape)

    sample_decoder_layer = TransformerDecoderBlock(512, 8, 2048)
    
    sample_decoder_layer_output, _, _ = sample_decoder_layer(tf.random.uniform((64, 50, 512)), sample_encoder_layer_output, None, None,False)

    print(sample_decoder_layer_output.shape)  # (batch_size, target_seq_len, d_model)

 


    