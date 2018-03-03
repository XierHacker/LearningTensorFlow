import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Prepare train data
train_X = np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.33 + 10

# Define the model
X = tf.placeholder("float")
Y = tf.placeholder("float")
w = tf.Variable(0.0, name="weight")
b = tf.Variable(0.0, name="bias")
loss = tf.square(Y - X*w - b)

#adjust learning rate
global_step=tf.Variable(initial_value=0,trainable=False)
start_learning_rate=0.1
learning_rate=tf.train.exponential_decay(
    learning_rate=start_learning_rate,
    global_step=global_step,
    decay_steps=1000,
    decay_rate=0.5,
    staircase=True
)

train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

# Create session to run
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    epoch = 1
    for i in range(11):
        for (x, y) in zip(train_X, train_Y):
            _, w_value, b_value = sess.run([train_op, w, b],feed_dict={X: x,Y: y})
            lr=sess.run([learning_rate,global_step])
        print("Epoch: {}, w: {}, b: {},lr:{}".format(epoch, w_value, b_value,lr))
        print("global step:",global_step)
        epoch += 1

#draw
plt.plot(train_X,train_Y,"+")
plt.plot(train_X,train_X.dot(w_value)+b_value)
plt.show()