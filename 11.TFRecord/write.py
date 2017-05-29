import tensorflow as tf
import numpy as np
import pandas as pd

#load .csv file
train_frame=pd.read_csv(filepath_or_buffer="train.csv")
#print(train_frame.head())
train_labels_frame=train_frame.pop(item="label")
print(train_labels_frame.shape)

train_values=train_frame.values
train_labels_values=train_labels_frame.values
train_size=train_values.shape[0]

'''
print(train_values[0].shape)
print(train_values[0].dtype)
print(train_labels_values[0])
'''


#------------------create TFRecord file------------------------#
writer=tf.python_io.TFRecordWriter(path="train.tfrecords")
for i in range(train_size):
    image_raw=train_values[i].tostring()

    example=tf.train.Example(
        features=tf.train.Features(
            feature={
                "image_raw":tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw])),
                "label":tf.train.Feature(int64_list=tf.train.Int64List(value=[train_labels_values[i]]))
            }
        )
    )
    writer.write(record=example.SerializeToString())

writer.close()
#tf.train.Feature()