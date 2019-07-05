import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt 


def draw_positional_encodings(PE):
    plt.pcolormesh(PE[0], cmap='RdBu')
    plt.xlabel('Depth')
    plt.xlim((0, 512))
    plt.ylabel('Position')
    plt.colorbar()
    plt.show()


def draw_multihead_attention(attention_weights,src_seq,tar_seq,layer):
    fig=plt.figure(figsize=(16,8))

    #sentence = tokenizer_pt.encode(sentence)
    attention_weights = tf.squeeze(attention_weights[layer], axis=0)

    for head in range(attention_weights.shape[0]):
        ax = fig.add_subplot(2, 4, head+1)

        # plot the attention weights
        ax.matshow(attention_weights[head][:-1, :], cmap='viridis')
        fontdict = {'fontsize': 10}
        ax.set_xticks(range(len(src_seq)+2))
        ax.set_yticks(range(len(tar_seq)))
        
        ax.set_ylim(len(tar_seq)-1.5, -0.5)
            
        ax.set_xticklabels(['<start>']+[i for i in src_seq]+['<end>'], fontdict=fontdict, rotation=90)
        
        ax.set_yticklabels([i for i in tar_seq], fontdict=fontdict)
        
        ax.set_xlabel('Head {}'.format(head+1))

    plt.tight_layout()
    plt.show()
    