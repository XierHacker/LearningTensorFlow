import tensorflow as tf

num_files=3
num_instance=100

for i in range(num_files):
    print("write ",i," file")
    fileName=("test.tfrecords-%.5d-of-%.5d" % (i,num_files))
    writer=tf.python_io.TFRecordWriter(path=fileName)

    for j in range(num_instance):
        print("write ",j," record")
        example=tf.train.Example(
            features=tf.train.Features(
                    feature={
                        "i":tf.train.Feature(int64_list=tf.train.Int64List(value=[i])),
                        "j":tf.train.Feature(int64_list=tf.train.Int64List(value=[j]))
                    }
            )

        )
        writer.write(record=example.SerializeToString())
    writer.close()

