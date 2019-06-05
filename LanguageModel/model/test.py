import numpy as np
import tensorflow as tf
import lstm
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
            "word":tf.io.VarLenFeature(dtype=tf.int64),
            "label":tf.io.VarLenFeature(dtype=tf.int64),
            "seq_len":tf.io.FixedLenFeature(shape=[], dtype=tf.int64)
        }
    )
    #get data,变长Feature会被处理为SparseTensor
    word=tf.cast(x=parsed_features["word"],dtype=tf.int32)
    label=tf.cast(x=parsed_features["label"],dtype=tf.int32)
    seq_len=tf.cast(x=parsed_features["seq_len"],dtype=tf.int32)
    return word,label,seq_len



def test(tfrecords_file_list):
    '''
    :param file_list:
    :return:
    '''
    model=lstm.LSTM_Model()

    #optimizer = tf.keras.optimizers.Adam(parameter.LEARNING_RATE)
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    checkpoint=tf.train.Checkpoint(model=model)

    checkpoint.restore(save_path=parameter.CHECKPOINT_PATH+"-1")


    # ----------------------------------------data set API-----------------------------------------
    # 创建dataset对象
    dataset = tf.data.TFRecordDataset(filenames=tfrecords_file_list)
    print("dataset:", dataset)
    # 使用map处理得到新的dataset
    parsed_dataset = dataset.map(map_func=_parse_data)
    parsed_dataset = parsed_dataset.batch(parameter.BATCH_SIZE).repeat(count=1)
    print("parsed_dataset:", parsed_dataset)
    # ----------------------------------------------------------------------------------------------
    iter_num=0
    for parsed_record in parsed_dataset:
        with tf.GradientTape() as tape:
            seq_len = parsed_record[2]
            mask = tf.sequence_mask(lengths=seq_len)

            # print("mask:\n",mask)
            word_dense=tf.sparse.to_dense(sp_input=parsed_record[0])
            #print("word_dense:\n",word_dense)
            label_dense=tf.sparse.to_dense(sp_input=parsed_record[1])
            label_onehot=tf.one_hot(indices=label_dense,depth=parameter.CLASS_NUM)
            label_onehot_masked=tf.boolean_mask(tensor=label_onehot,mask=mask,axis=0)
            #print("label_onehot_masked.shape:\n",label_onehot_masked.shape)

            logits=model(word_ids=word_dense,mask=mask,training=True)
            #print("logits.shape:\n",logits)

            logits_masked=tf.boolean_mask(tensor=logits,mask=mask,axis=0)
            #print("logits_mask.shape:\n",logits_masked.shape)

            loss=cce(y_true=label_onehot_masked,y_pred=logits_masked)

            print("loss:",loss.numpy())
            print("perplexity:",np.exp(loss.numpy()))

            # 计算梯度
            #gradient = tape.gradient(target=loss, sources=model.trainable_variables)
            # print("gradient:",gradient)
            # 应用梯度
            #optimizer.apply_gradients(zip(gradient, model.trainable_variables))

        iter_num+=1

        #save checkpoints every 200 iterations
        #if iter_num%500==0:
        #    checkpoint.save(file_prefix=parameter.CHECKPOINT_PATH)




if __name__=="__main__":
    test(tfrecords_file_list=parameter.TEST_FILE_LIST)