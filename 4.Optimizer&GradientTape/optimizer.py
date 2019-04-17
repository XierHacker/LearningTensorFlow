import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

TRAIN_STEPS=20

# Prepare train data
train_X = np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.33 + 10

print(train_X.shape)

w=tf.Variable(initial_value=1.0)
b=tf.Variable(initial_value=1.0)

optimizer=tf.keras.optimizers.SGD(0.1)
mse=tf.keras.losses.MeanSquaredError()

for i in range(TRAIN_STEPS):
    print("epoch:",i)
    print("w:", w.numpy())
    print("b:", b.numpy())
    #计算和更新梯度
    with tf.GradientTape() as tape:
        logit = w * train_X + b
        loss=mse(train_Y,logit)
    gradients=tape.gradient(target=loss,sources=[w,b])  #计算梯度
    #print("gradients:",gradients)
    #print("zip:\n",list(zip(gradients,[w,b])))
    optimizer.apply_gradients(zip(gradients,[w,b]))     #更新梯度


#draw
plt.plot(train_X,train_Y,"+")
plt.plot(train_X,w * train_X + b)
plt.show()