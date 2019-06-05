import numpy as np
import pandas as pd
import tensorflow as tf
import model
import parameter

MAX_TIME_TARGET=80


def _parse_data(example_proto):
    '''
    定义tfrecords解析和预处理函数
    :param example_proto:
    :return:
    '''
    parsed_features = tf.io.parse_single_example(
        serialized=example_proto,
        features={
            "src_word":tf.io.VarLenFeature(dtype=tf.int64),
            "src_len": tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
            "target_word_input": tf.io.VarLenFeature(dtype=tf.int64),
            "target_word_output": tf.io.VarLenFeature(dtype=tf.int64),
            "target_len":tf.io.FixedLenFeature(shape=[], dtype=tf.int64)
        }
    )

    #变长Feature会被处理为SparseTensor
    src_word=tf.cast(x=parsed_features["src_word"],dtype=tf.int32)
    src_len = tf.cast(x=parsed_features["src_len"], dtype=tf.int32)
    target_word_input=tf.cast(x=parsed_features["target_word_input"],dtype=tf.int32)
    target_word_output = tf.cast(x=parsed_features["target_word_output"], dtype=tf.int32)
    target_len=tf.cast(x=parsed_features["target_len"],dtype=tf.int32)
    return src_word,src_len,target_word_input,target_word_output,target_len

def getWordsMapper(IndexFile):
    df_words_ids = pd.read_csv(filepath_or_buffer=IndexFile, encoding="utf-8")
    words2id = pd.Series(data=df_words_ids["id"].values, index=df_words_ids["word"].values)
    id2words = pd.Series(data=df_words_ids["word"].values, index=df_words_ids["id"].values)
    # print("word2id:\n",words2id)
    # print("id2words:\n",id2words)
    # print("words2id.shape",words2id.shape)
    return words2id,id2words


def loss_func(loss_obj,real,pred):
    '''
    精细的损失函数
    :param loss_obj:损失函数对象tf.keras.losses.xxx
    :param real: 真实标签，形状为[batch_size,]
    :param pred: 预测标签，这里可以是logits，形状为[batch_size,class_num]
    :return:
    '''
    #print("real:\n",real)
    mask=tf.math.logical_not(tf.math.equal(x=real,y=0))
    #print("mask:\n",mask)

    loss=loss_obj(real,pred)
    #print("loss:",loss)

    mask=tf.cast(x=mask,dtype=loss.dtype)
    loss*=mask
    #print("loss:", loss)

    loss=tf.reduce_mean(loss)
    #print("loss:", loss)
    #print("\n\n")

    return loss



def test(tfrecords_file_list):
    '''
    :param file_list:
    :return:
    '''

    words2id_zh, id2words_zh = getWordsMapper("../index_files/zh_ids.csv")
    words2id_en, id2words_en = getWordsMapper("../index_files/en_ids.csv")
    encoder=model.Encoder(
        vocab_size=parameter.SRC_VOCAB_SIZE,
        embeddings_dim=parameter.EMBEDDINGS_DIM,
        units=128,
        batch_size=parameter.BATCH_SIZE
    )

    decoder=model.Decoder(
        vocab_size=parameter.TARGET_VOCAB_SIZE,
        embeddings_dim=parameter.EMBEDDINGS_DIM,
        units=128,
        batch_size=parameter.BATCH_SIZE
    )

    optimizer = tf.keras.optimizers.Adam(parameter.LEARNING_RATE)
    cce=tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction=tf.keras.losses.Reduction.NONE
    )
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,encoder=encoder, decoder=decoder)

    #restore
    checkpoint.restore(save_path=parameter.CHECKPOINT_PATH+"-1")

    # ----------------------------------------data set API-----------------------------------------
    # 创建dataset对象
    dataset = tf.data.TFRecordDataset(filenames=tfrecords_file_list)
    print("dataset:", dataset)
    # 使用map处理得到新的dataset
    parsed_dataset = dataset.map(map_func=_parse_data)
    parsed_dataset = parsed_dataset.batch(1).repeat(1)  #batch_size只为1
    print("parsed_dataset:", parsed_dataset)
    # ----------------------------------------------------------------------------------------------
    iter_num=0      #迭代次数
    for parsed_record in parsed_dataset:            #一次一个mini_batch
        #准备数据,这里只取源序列，忽略目标序列
        src_word=tf.sparse.to_dense(parsed_record[0])
        #print("src_word:",src_word[0].numpy())
        src_len=parsed_record[1]
        #target_word_input=tf.sparse.to_dense(parsed_record[2])
        #target_word_output=tf.sparse.to_dense(parsed_record[3])
        #target_len=parsed_record[4]

        src_mask = tf.sequence_mask(lengths=src_len)
        #print("src_mask:\n",src_mask)

        #encode
        en_outputs, en_states = encoder(word_ids=src_word, mask=src_mask, training=False)
        # print("en_outputs:\n",en_outputs)
        # print("en_states:\n",en_states)

        #decode
        pre_states=en_states        #decoder的第一个state设置为encoder输出的那个states

        target_word_input_one_step=tf.expand_dims(input=[1],axis=0) #start flag <sos>
        #print("target_word_input_one_step:", target_word_input_one_step)

        #receive result
        predict_ids=[]
        for time in range(MAX_TIME_TARGET):
            de_outputs,de_states,attention_weights=decoder(
                word_ids=target_word_input_one_step,
                pre_states=pre_states,
                encoder_outputs=en_outputs
            )
            #loss+=loss_func(loss_obj=cce,real=target_word_output_one_step,pred=de_outputs)
            pre_states=de_states    #重新赋值states

            predict_id=tf.argmax(input=de_outputs[0]).numpy()

            if predict_id==2:
                break
            predict_ids.append(predict_id)
        recover(src_ids=src_word[0].numpy(),pred_ids=predict_ids,src_mapper=id2words_zh,target_mapper=id2words_en)


def recover(src_ids,pred_ids,src_mapper,target_mapper):
    '''
    从id序列中恢复为文本
    :param src_ids: 源单词id序列
    :param pred_ids: 预测结果id序列
    :param src_mapper: 源索引mapper
    :param target_mapper: 预测索引mapper
    :return:
    '''
    src_text=""
    pred_text=""

    for src_id in src_ids:
        src_text+=src_mapper[src_id]+" "

    for pred_id in pred_ids:
        pred_text+=target_mapper[pred_id]+" "

    print("src_text:",src_text)
    print("pred_text:",pred_text)
    print("\n\n")



if __name__=="__main__":
    test(tfrecords_file_list=parameter.TRAIN_FILE_LIST)