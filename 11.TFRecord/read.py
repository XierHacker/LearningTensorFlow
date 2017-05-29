import tensorflow as tf
import numpy as np

graph=tf.Graph()
session=tf.Session(graph=graph)

with graph.as_default():
    files=tf.train.match_filenames_once(pattern="train.tfrecords").initialized_value()
    #print("files:",files)
    filename_queue=tf.train.string_input_producer(string_tensor=files,shuffle=False)

    reader=tf.TFRecordReader()
    _,serialized_example=reader.read(queue=filename_queue)

    features=tf.parse_single_example(
            serialized=serialized_example,
            features={
                "image_raw": tf.FixedLenFeature([],tf.string),
                "label": tf.FixedLenFeature([],tf.int64)
            }
        )

    #get single feature
    raw=features["image_raw"]
    label = features["label"]
    # decode raw
    image = tf.decode_raw(bytes=raw, out_type=tf.int64)

    init_op=tf.global_variables_initializer()

with session.as_default():
    session.run(init_op)
    print(session.run(files))

    #threads
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(sess=session,coord=coord)

    image_run,label_run=session.run(fetches=[image,label])
    print(type(image_run))
    print(image_run.shape)
    print(image_run)
    print(label_run)


    coord.request_stop()
    coord.join(threads)



