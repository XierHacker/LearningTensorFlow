EMBEDDINGS_DIM=200

SRC_VOCAB_SIZE=5029+1
TARGET_VOCAB_SIZE=20003+1
CLASS_NUM=TARGET_VOCAB_SIZE

DROP_OUT_RATE=0.2
HIDDEN_UNITS=128

MAX_EPOCH=10
BATCH_SIZE=30
LEARNING_RATE=0.001


TRAIN_FILE_LIST=["../dataset/wmt_train.tfrecords"]

#TRAIN_SIZE=statistic.getTFRecordsListAmount(tfFileList=TRAIN_FILE_LIST)

#测试集样本数目
TEST_FILE_LIST=["../dataset/wmt_train.tfrecords"]
#TEST_SIZE=statistic.getTFRecordsListAmount(tfFileList=TEST_FILE_LIST)


#模型存放地址
CHECKPOINT_PATH="./checkpoints/lstm"

#可视化log存放地址
LOG_DIR="./log/"
