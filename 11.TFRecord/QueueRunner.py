import tensorflow as tf
import numpy as np

graph=tf.Graph()
session=tf.Session(graph=graph)

with graph.as_default():
    #create a FIFO queue
    queue=tf.FIFOQueue(capacity=10,dtypes=tf.float32)
    #enqueue Op
    enqueue_op=queue.enqueue(vals=tf.random_normal(shape=(2,2)))

    #create a QueueRunner object
    qr=tf.train.QueueRunner(queue=queue,enqueue_ops=[enqueue_op]*5)

    #add qr to tf.GraphKeys.QUEUE_RUNNERS
    tf.train.add_queue_runner(qr=qr)

    #size1
  #  size1=queue.size()

    #dequeue op
    out_tensor=queue.dequeue()

with session.as_default():
    #use Coordinator handle threads
    coord=tf.train.Coordinator()

    #start queue_runner and get a list of threads
    threads=tf.train.start_queue_runners(sess=session,coord=coord)

    #size1
    #print("size1 of queue:",session.run(queue.size()))

    for i in range(3):
        print("out tensor:",session.run(out_tensor))


    coord.request_stop()
    coord.join(threads)