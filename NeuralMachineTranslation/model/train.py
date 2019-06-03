import numpy as np
import tensorflow as tf
import model
import parameter

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



def train(tfrecords_file_list):
    '''
    :param file_list:
    :return:
    '''
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
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)

    #log
    file_writer=tf.summary.create_file_writer(logdir=parameter.LOG_DIR)

    # ----------------------------------------data set API-----------------------------------------
    # 创建dataset对象
    dataset = tf.data.TFRecordDataset(filenames=tfrecords_file_list)
    print("dataset:", dataset)
    # 使用map处理得到新的dataset
    parsed_dataset = dataset.map(map_func=_parse_data)
    parsed_dataset = parsed_dataset.shuffle(buffer_size=1000).batch(parameter.BATCH_SIZE).repeat(count=parameter.MAX_EPOCH)
    print("parsed_dataset:", parsed_dataset)
    # ----------------------------------------------------------------------------------------------
    iter_num=0      #迭代次数
    for parsed_record in parsed_dataset:            #一次一个mini_batch
        with tf.GradientTape() as tape:
            #准备数据
            src_word=tf.sparse.to_dense(parsed_record[0])
            src_len=parsed_record[1]
            target_word_input=tf.sparse.to_dense(parsed_record[2])
            target_word_output=tf.sparse.to_dense(parsed_record[3])
            target_len=parsed_record[4]
            # print("src_words:", src_word)
            # print("src_len", src_len)
            # print("target_word_input:", target_word_input)
            # print("target_word_output:", target_word_output)
            #print("target_word_input.shape[1]:",target_word_input.shape[1])
            #print("target_len", parsed_record[4])
            # print("\n\n")

            #mini batch loss
            loss=0

            src_mask = tf.sequence_mask(lengths=src_len)
            #print("src_mask:\n",src_mask)

            #encode
            en_outputs, en_states = encoder(word_ids=src_word, mask=src_mask, training=True)
            # print("en_outputs:\n",en_outputs)
            # print("en_states:\n",en_states)

            pre_states=en_states        #decoder的第一个state设置为encoder输出的那个states
            for time in range(target_word_input.shape[1]):
                target_word_input_one_step = target_word_input[:, time]
                target_word_input_one_step = tf.expand_dims(input=target_word_input_one_step, axis=-1)
                #print("target_word_input_one_step:", target_word_input_one_step)
                target_word_output_one_step=target_word_output[:,time]
                #print("target_word_output_one_step:", target_word_output_one_step)
                de_outputs,de_states,attention_weights=decoder(
                    word_ids=target_word_input_one_step,
                    pre_states=pre_states,
                    encoder_outputs=en_outputs
                )
                loss+=loss_func(loss_obj=cce,real=target_word_output_one_step,pred=de_outputs)
                pre_states=de_states    #重新赋值states

            #这里可以loss除以时间步
            loss=loss/target_word_input.shape[1]
            print("loss:",loss)

            #添加标量到summary
            with file_writer.as_default():
                tf.summary.scalar(name="loss",data=loss,step=iter_num)
                file_writer.flush()


            #optimize
            variables=encoder.trainable_variables+decoder.trainable_variables
            gradients=tape.gradient(target=loss,sources=variables)
            optimizer.apply_gradients(zip(gradients,variables))

        iter_num+=1
    file_writer.close()


if __name__=="__main__":
    train(tfrecords_file_list=parameter.TRAIN_FILE_LIST)