from __future__ import print_function

import numpy as np
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


dat = np.load('outfile.npy') 
train_Y = np.array(dat[:,3])
train_X1 = np.array(dat[:,0])
train_X2 = np.array(dat[:,1])
train_X3 = np.array(dat[:,2])
rng = np.random

# Parameters
learning_rate = 0.01
training_epochs = 1000
display_step = 50

n_samples = train_X1.shape[0]


# tf Graph Input
X1 = tf.placeholder("float")
X2 = tf.placeholder("float")
X3 = tf.placeholder("float")
Y = tf.placeholder("float")


# Set model weights
W1 = tf.Variable(rng.randn(), name="weight1")
W2 = tf.Variable(rng.randn(), name="weight2")
W3 = tf.Variable(rng.randn(), name="weight3")
b = tf.Variable(rng.randn(), name="bias")

# Construct a linear model
pred = tf.add(tf.add(tf.multiply(X1, W1), tf.multiply(X2, W2)), tf.add(tf.multiply(X3, W3), b))

# Mean squared error
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)

#  Gradient descent
#  Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        for (x1, x2, x3, y) in zip(train_X1, train_X2, train_X3, train_Y):
            sess.run(optimizer, feed_dict={X1: x1, X2: x2, X3: x3, Y: y})

        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X1: train_X1, X2: train_X2, X3: train_X3, Y:train_Y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "W1=", sess.run(W1), "W2=", sess.run(W2), "W3=", sess.run(W3), "b=", sess.run(b))

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X1: train_X1, X2: train_X2, X3: train_X3, Y: train_Y})
    print("Training cost=", training_cost, "W1=", sess.run(W1), "W2=", sess.run(W2), "W3=", sess.run(W3), "b=", sess.run(b), '\n')

    # Graphic display
# plt.plot(train_X1, train_Y, 'ro', label='Original data')
# plt.plot(train_X1, sess.run(W2) * train_X2 + sess.run(b), label='Fitted line')
# plt.legend()
# plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(train_X2, train_X3, train_Y, c='b', marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

print('')
