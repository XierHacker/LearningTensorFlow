import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


#tfrecord 文件列表
file_list=["train.tfrecords"]

#创建dataset对象
dataset=tf.data.TFRecordDataset(filenames=file_list)

#定义解析和预处理函数
def _parse_data(example_proto):
    parsed_features=tf.parse_single_example(
        serialized=example_proto,
        features={
            "image_raw":tf.FixedLenFeature(shape=(),dtype=tf.string),
            "label":tf.FixedLenFeature(shape=(),dtype=tf.int64)
        }
    )

    # get single feature
    raw = parsed_features["image_raw"]
    label = parsed_features["label"]
    # decode raw
    image = tf.decode_raw(bytes=raw, out_type=tf.int64)
    image=tf.reshape(tensor=image,shape=[28,28])
    return image,label

#使用map处理得到新的dataset
dataset=dataset.map(map_func=_parse_data)
#dataset = dataset.batch(32)

#创建迭代器
iterator=dataset.make_one_shot_iterator()

next_element=iterator.get_next()

with tf.Session() as sess:
    for i in range(10):
        image, label = sess.run(next_element)
        print(label)
        print(image.shape)
        print(label.shape)
        #plt.imshow(image)
        #plt.show()



