import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

lables=np.array([0,1,0,1,1,0,1,0,0,0])
pred=np.array([0,1,0,0,1,0,0,0,0,0])

f_score=f1_score(y_true=lables,y_pred=pred)
print("sklearn_based f1 score:",f_score)

labels_t=tf.constant(value=lables)
pred_t=tf.constant(value=pred)


# correct_prediction
#correct_prediction = tf.equal(labels_t,pred_t)

# accracy
#accuracy_pw = tf.reduce_mean(
#        input_tensor=tf.cast(x=correct_prediction, dtype=tf.float32),
#        name="accuracy_pw"
#)

p=tf.metrics.precision(labels=labels_t,predictions=pred_t,name="precision")[1]
r=tf.metrics.recall(labels=labels_t,predictions=pred_t,name="recall")[1]
f1=2*p*r/(r+p)
local_init=tf.local_variables_initializer()
global_init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(global_init)
    sess.run(local_init)
    print("precision:",sess.run(p))
    print("recall:",sess.run(r))
    print("f1 score:",sess.run(f1))
    #print(sess.run(accu))