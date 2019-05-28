

EMBEDDINGS_DIM=200
VOCAB_SIZE=10001+1
CLASS_NUM=10001+1
DROP_OUT_RATE=0.2
HIDDEN_UNITS=128

MAX_EPOCH=10
BATCH_SIZE=30
LEARNING_RATE=0.001



TRAIN_FILE_LIST=["../dataset/train.tfrecords"]

#TRAIN_SIZE=statistic.getTFRecordsListAmount(tfFileList=TRAIN_FILE_LIST)

#测试集样本数目
TEST_FILE_LIST=["../dataset/test.tfrecords"]
#TEST_SIZE=statistic.getTFRecordsListAmount(tfFileList=TEST_FILE_LIST)

