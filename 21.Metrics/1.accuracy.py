import numpy as np
import tensorflow as tf


lables=np.array([1,2,3,4,1,1,1,3,4,1])
pred=np.array([1,2,3,4,1,1,1,3,4,0])



labels_t=tf.constant(value=lables)
pred_t=tf.constant(value=pred)


# correct_prediction
correct_prediction = tf.equal(labels_t,pred_t)

# accracy
accuracy_pw = tf.reduce_mean(
        input_tensor=tf.cast(x=correct_prediction, dtype=tf.float32),
        name="accuracy_pw"
)

accu=tf.metrics.accuracy(labels=labels_t,predictions=pred_t,name="accu")[1]
local_init=tf.local_variables_initializer()
global_init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(global_init)
    sess.run(local_init)
    print(sess.run(accuracy_pw))
    print(sess.run(accu))