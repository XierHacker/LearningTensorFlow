import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


#"../../data/DogsVsCats/train/cat.0.jpg"
#0~12499

def pics2tfrecords(folder,is_train):
    '''
    :param folder:  folder to storage pics
    :param is_train:train set of validation set
    :return:
    '''
    print("Trans Pictures To TFRecords")
    if is_train:
        kind_cat = 0
        kind_dog = 1
        writer_train = tf.python_io.TFRecordWriter(path="dog_vs_cat_train.tfrecords")
        writer_valid = tf.python_io.TFRecordWriter(path="dog_vs_cat_valid.tfrecords")

        #training set
        for i in range(10):
            pic_cat=cv2.imread(filename=folder+"train/cat."+str(i)+".jpg",flags=cv2.IMREAD_UNCHANGED)
            #resize to 250x250
            pic_cat=cv2.resize(src=pic_cat,dsize=(250,250),interpolation=cv2.INTER_AREA)
            #to string
            pic_cat_raw=pic_cat.tostring()
            print(pic_cat.shape)

            example=tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "image_raw":tf.train.Feature(bytes_list=tf.train.BytesList(value=[pic_cat_raw])),
                        "label":tf.train.Feature(int64_list=tf.train.Int64List(value=[kind_cat]))
                    }
                )
            )
            writer_train.write(record=example.SerializeToString())

            pic_dog = cv2.imread(filename=folder + "train/dog." + str(i) + ".jpg", flags=cv2.IMREAD_UNCHANGED)
            # resize to 250x250
            pic_dog = cv2.resize(src=pic_dog, dsize=(250, 250), interpolation=cv2.INTER_AREA)
            # to string
            pic_dog_raw = pic_dog.tostring()
            print(pic_dog.shape)

            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "image_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[pic_dog_raw])),
                        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[kind_dog]))
                    }
                )
            )
            writer_train.write(record=example.SerializeToString())

        writer_train.close()

        #validation set
        for i in range(10,20):
            pic_cat = cv2.imread(filename=folder + "train/cat." + str(i) + ".jpg", flags=cv2.IMREAD_UNCHANGED)
            # resize to 300x300
            pic_cat = cv2.resize(src=pic_cat, dsize=(250, 250), interpolation=cv2.INTER_AREA)
            # to string
            pic_cat_raw = pic_cat.tostring()
            print(pic_cat.shape)

            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "image_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[pic_cat_raw])),
                        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[kind_cat]))
                    }
                )
            )
            writer_valid.write(record=example.SerializeToString())

            pic_dog = cv2.imread(filename=folder + "train/dog." + str(i) + ".jpg", flags=cv2.IMREAD_UNCHANGED)
            # resize to 300x300
            pic_dog = cv2.resize(src=pic_dog, dsize=(250, 250), interpolation=cv2.INTER_AREA)
            # to string
            pic_dog_raw = pic_dog.tostring()
            print(pic_dog.shape)

            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "image_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[pic_dog_raw])),
                        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[kind_dog]))
                    }
                )
            )
            writer_valid.write(record=example.SerializeToString())

        writer_valid.close()

    else:
        pass


if __name__=="__main__":
    #pic = mpimg.imread(fname="../CIFAR-10/train/2.png")
    #print(pic.shape)
    #train_labels_frame=pd.read_csv(filepath_or_buffer="../CIFAR-10/trainLabels.csv")
    #print(train_labels_frame)
    #print(train_labels_frame[train_labels_frame['id']==2])
    #plt.imshow(pic)
    #plt.show()
    #print(maping_dict["dog"])
    #print(train_labels_frame["label"][0])

    pics2tfrecords(folder="../../data/DogsVsCats/",is_train=True)