import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#mapping name to number
mapping_dict={
    "frog":0,
    "truck":1,
    "deer":2,
    "automobile":3,
    "bird":4,
    "horse":5,
    "ship":6,
    "cat":7,
    "airplane":8,
    "dog":9
}


def pics2tfrecords(folder,is_train):
    '''
    :param folder:  folder to storage pics
    :param is_train:train set of validation set
    :return:
    '''
    print("Trans Pictures To TFRecords")
    if is_train:
        train_labels_frame = pd.read_csv(filepath_or_buffer=folder+"trainLabels.csv")
        #training set
        writer_train=tf.python_io.TFRecordWriter(path="cifar_10_train.tfrecords")
        for i in range(1,45000+1):
            pic = mpimg.imread(fname=folder+"train/"+str(i)+".png")
            pic_raw=pic.tostring()
            kind=mapping_dict[train_labels_frame["label"][i-1]]

            example=tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "image_raw":tf.train.Feature(bytes_list=tf.train.BytesList(value=[pic_raw])),
                        "label":tf.train.Feature(int64_list=tf.train.Int64List(value=[kind]))
                    }
                )
            )
            writer_train.write(record=example.SerializeToString())
        writer_train.close()

        #validation set
        #for i in range(45000+1,50000+1):
        #    pic = mpimg.imread(fname=folder + "train/" + str(i) + ".png")


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

    pics2tfrecords(folder="../CIFAR-10/",is_train=True)


