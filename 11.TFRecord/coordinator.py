import tensorflow as tf
import numpy as np
import threading
import time

def loop(coord,id):
    while not coord.should_stop():
        if np.random.rand()<0.01:
            print("when in id: ",id,", request stop")
            coord.request_stop()
        else:
            print("this is id:",id)

        time.sleep(1)

#create coordinator
coord=tf.train.Coordinator()

threads=[threading.Thread(target=loop,args=(coord,i)) for i in range(5)]

for thread in threads:
    thread.start()

coord.join(threads)


