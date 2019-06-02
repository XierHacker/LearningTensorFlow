import numpy as np
import pandas as pd
import tensorflow as tf

IN_FILE_ZH="../wmt_corpus/processed.zh"
IN_FILE_EN="../wmt_corpus/processed.en"
OUT_FILE="wmt_train.tfrecords"

write=True

def getWordsMapper(IndexFile):
    df_words_ids = pd.read_csv(filepath_or_buffer=IndexFile, encoding="utf-8")
    words2id = pd.Series(data=df_words_ids["id"].values, index=df_words_ids["word"].values)
    id2words = pd.Series(data=df_words_ids["word"].values, index=df_words_ids["id"].values)
    # print("word2id:\n",words2id)
    # print("id2words:\n",id2words)
    # print("words2id.shape",words2id.shape)
    return words2id,id2words



def index(src_word_list,target_word_list,src_mapper,target_mapper):
    '''
    把源语言句子和目标语言句子转换为索引，并且返回真实长度
    :param src_word_list: 源语言句子序列
    :param target_word_list: 目标语言句子序列
    :param src_mapper: 源语言的mapper
    :param target_mapper: 目标语言的mapper
    :return:
    '''
    src_index=[]
    target_index_inputs=[]
    target_index_inputs.append(target_mapper["<sos>"])     #在开始添加开始"<sos>"标记
    target_index_outputs=[]
    seq_len_src=0
    seq_len_target=0
    #handle src word list
    for word in src_word_list:
        src_index.append(src_mapper[word])

    print("src_index:",src_index)

    # handle target word list
    for word in target_word_list:
        target_index_inputs.append(target_mapper[word])
        target_index_outputs.append(target_mapper[word])
    target_index_outputs.append(target_mapper["<eos>"])     #在末尾添加结束"<eos>"标记

    print("target_index_inputs:", target_index_inputs,len(target_index_inputs))
    print("target_index_outputs:", target_index_outputs,len(target_index_outputs))
    print("\n\n")


    # print("target_index:", target_index)




def preprocess(infile_zh,infile_en,outfile):
    '''
    把原始文本处理为tfrecords
    :param infile_zh: 输入的中文文件
    :param infile_en: 输入的英文文件
    :param outfile: 输出的tfrecords
    :return:
    '''
    words2id_zh, id2words_zh=getWordsMapper("../index_files/zh_ids.csv")
    words2id_en, id2words_en = getWordsMapper("../index_files/en_ids.csv")
    # print("word2id:",words2id_en)
    # print("id2word:",id2words_en)
    file_zh=open(file=infile_zh,encoding="utf-8",errors="ignore")
    lines_zh=file_zh.readlines()
    file_en=open(file=infile_en,encoding="utf-8",errors="ignore")
    lines_en=file_en.readlines()

    # print("lines_zh:\n",lines_zh)
    # print("lines_en:\n",lines_en)

    if len(lines_en)!=len(lines_zh):
        print("wrong!!1")
        return None

    for i in range(len(lines_zh)):
        line_zh=lines_zh[i].strip()
        line_en=lines_en[i].strip()
        # 丢掉所有空句子
        if line_zh == "" or line_en == "":
            print("line_zh:", line_zh)
            print("line_en:", line_en)
            continue
        #丢掉所有<开头的句子
        if line_zh[0]=="<":
            continue
        word_list_zh = line_zh.split(sep=" ")
        word_list_en = line_en.split(sep=" ")
        #丢掉长度超过80的句子
        if len(word_list_zh)>80 or len(word_list_en)>80:
            print("line_zh:", word_list_zh)
            print("line_en:", word_list_en)
            continue

        # print("line_zh:",word_list_zh)
        # print("line_en:",word_list_en)
        # index(src_word_list=word_list_zh,target_word_list=word_list_en,src_mapper=words2id_zh,target_mapper=words2id_en)






    file_zh.close()
    file_en.close()








    # writer=tf.io.TFRecordWriter(path=outfile)
    # #print("word2id:",words2id)
    # with open(file=infile,encoding="utf-8",errors="ignore") as in_file:
    #     lines=in_file.readlines()
    #     # print("lines:\n",lines)
    #     for line in lines:
    #         word_list=line.strip().split(sep=" ")
    #         #print("word_list:",word_list)
    #         x,y,seq_len=index(word_list=word_list,mapper=words2id)
    #         # 写入到tfrecords
    #         example = tf.train.Example(
    #             features=tf.train.Features(
    #                 feature={
    #                     "word": tf.train.Feature(int64_list=tf.train.Int64List(value=x)),
    #                     "label": tf.train.Feature(int64_list=tf.train.Int64List(value=y)),
    #                     "seq_len": tf.train.Feature(int64_list=tf.train.Int64List(value=[seq_len]))
    #                 }
    #             )
    #         )
    #         writer.write(record=example.SerializeToString())
    #     writer.close()



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
    # 使用map处理得到新的dataset
    parsed_dataset = dataset.map(map_func=_parse_data)
    # dataset = dataset.map(map_func=_parse_data)
    parsed_dataset = parsed_dataset.batch(2)
    print("parsed_dataset:", parsed_dataset)
    #for parsed_record in parsed_dataset.take(1):
    #    print("parsed_records:",parsed_record[0])
    for parsed_record in parsed_dataset:
        print("parsed_records:", parsed_record[1])


if __name__=="__main__":
    print(IN_FILE_ZH)
    print(IN_FILE_EN)
    print(OUT_FILE)
    if write:
        preprocess(infile_en=IN_FILE_EN,infile_zh=IN_FILE_ZH,outfile=OUT_FILE)
    else:
        readTFRecords(tfrecords_file_list=["train.tfrecords"])

