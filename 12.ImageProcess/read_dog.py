import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#tfrecord 文件列表
file_list=["dog_vs_cat_valid.tfrecords"]

#创建dataset对象
dataset=tf.data.TFRecordDataset(filenames=file_list)

#定义解析和预处理函数
def _parse_data(example_proto):
    parsed_features=tf.parse_single_example(
        serialized=example_proto,
        features={
            "image_raw":tf.FixedLenFeature(shape=[],dtype=tf.string),
            "label":tf.FixedLenFeature(shape=[],dtype=tf.int64)
        }
    )

    # get single feature
    raw = parsed_features["image_raw"]
    label = parsed_features["label"]
    # decode raw
    image = tf.decode_raw(bytes=raw, out_type=tf.uint8)
    image = tf.reshape(tensor=image, shape=(250, 250, 3))
    #crop
    image=tf.image.resize_image_with_crop_or_pad(image=image,target_height=224,target_width=224)
    #trans to float
    image=tf.image.convert_image_dtype(image=image,dtype=tf.float32)
    return image,label

#使用map处理得到新的dataset
dataset=dataset.map(map_func=_parse_data)
#使用batch_size为32生成mini-batch
#dataset = dataset.batch(32)

#创建迭代器
iterator=dataset.make_one_shot_iterator()

next_element=iterator.get_next()

with tf.Session() as sess:
    for i in range(10):
        element = sess.run(next_element)
        print("label:",element[1])
        print("image.shape:",element[0].shape)
        print("label.shape",element[1].shape)

        plt.imshow(element[0])
        plt.show()



#if __name__=="__main__":
