import tensorflow as tf

graph=tf.Graph()
session=tf.Session(graph=graph)
with graph.as_default():
    files=tf.train.match_filenames_once(pattern="test.tfrecords-*").initialized_value()
    #print(type(files))

    filename_queue=tf.train.string_input_producer(string_tensor=files,shuffle=False)
    size=filename_queue.size()
    reader=tf.TFRecordReader()
    _,serialized_example=reader.read(queue=filename_queue)

    features=tf.parse_single_example(
                serialized=serialized_example,
                features={
                    "i":tf.FixedLenFeature([],tf.int64),
                    "j":tf.FixedLenFeature([],tf.int64)

                }
        )

    init_op=tf.global_variables_initializer()


with session.as_default():
    session.run(init_op)
    files_run=session.run(files)
    print(files_run)
    print(type(files_run))
    # threads
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=session, coord=coord)

    while True:
        print(session.run(fetches=[features["i"],features["j"]]))
        print("size of queue:",session.run(size))

    coord.request_stop()
    coord.join(threads)