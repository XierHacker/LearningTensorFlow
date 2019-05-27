import numpy as np
import pandas as pd
import tensorflow as tf

MODE="train"

IN_FILE="../ptb_corpus/ptb."+MODE+".txt"
OUT_FILE="./"+MODE+".tfrecords"

write=False

def getWordsMapper(IndexFile):
    df_words_ids = pd.read_csv(filepath_or_buffer=IndexFile, encoding="utf-8")
    words2id = pd.Series(data=df_words_ids["id"].values, index=df_words_ids["word"].values)
    id2words = pd.Series(data=df_words_ids["word"].values, index=df_words_ids["id"].values)
    # print("word2id:\n",words2id)
    # print("id2words:\n",id2words)
    # print("words2id.shape",words2id.shape)
    return words2id,id2words



def index(word_list,mapper):
    '''
    :param word_list:
    :param mapper:
    :return:
    '''
    #print("word_list:",word_list)
    x=[]
    y=[]
    for i in range(len(word_list)):
        if i==len(word_list)-1:
            x.append(mapper[word_list[i]])
            y.append(mapper["<eos>"])
            break
        else:
            x.append(mapper[word_list[i]])
            y.append(mapper[word_list[i+1]])
    seq_len = len(x)
    # print("x:\n",x)
    # print("y:\n",y)
    # print("\n\n")
    return x,y,seq_len



def preprocess(infile,outfile):
    '''

    :param infile:
    :param outfile:
    :return:
    '''
    words2id, id2words=getWordsMapper("../index_files/words_ids.csv")
    writer=tf.io.TFRecordWriter(path=outfile)
    #print("word2id:",words2id)
    with open(file=infile,encoding="utf-8",errors="ignore") as in_file:
        lines=in_file.readlines()
        # print("lines:\n",lines)
        for line in lines:
            word_list=line.strip().split(sep=" ")
            #print("word_list:",word_list)
            x,y,seq_len=index(word_list=word_list,mapper=words2id)
            # 写入到tfrecords
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "word": tf.train.Feature(int64_list=tf.train.Int64List(value=x)),
                        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=y)),
                        "seq_len": tf.train.Feature(int64_list=tf.train.Int64List(value=[seq_len]))
                    }
                )
            )
            writer.write(record=example.SerializeToString())
        writer.close()



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


def readTFRecords(tfrecords_file_list):
    '''
    读取tfrecords中的内容
    :param inFile: tfrecords
    :return:
    '''
    # ----------------------------------------data set API-----------------------------------------
    # 创建dataset对象
    dataset = tf.data.TFRecordDataset(filenames=tfrecords_file_list)
    print("dataset:",dataset)
    # # # 使用map处理得到新的dataset
    parsed_dataset = dataset.map(map_func=_parse_data)
    # dataset = dataset.map(map_func=_parse_data)
    parsed_dataset = parsed_dataset.batch(2)

    for parsed_record in parsed_dataset.take(1):
        print("parsed_records:",parsed_record[0])


if __name__=="__main__":
    print(IN_FILE)
    print(OUT_FILE)
    if write:
        preprocess(infile=IN_FILE,outfile=OUT_FILE)
    else:
        readTFRecords(tfrecords_file_list=["train.tfrecords"])

