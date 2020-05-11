# It introduces basic variables and functions
# and shows how you can optimize a function.


import numpy as np
import tensorflow as tf


# you have to specify the type
A = tf.placeholder(tf.float32, shape=(5, 5), name='A')

# but shape and name are optional
v = tf.placeholder(tf.float32)

# I think this name is more appropriate than 'dot'
w = tf.matmul(A, v)

# similar to Theano, you need to "feed" the variables values.
# In TensorFlow you do the "actual work" in a "session"!

with tf.Session() as session:
    # the values are fed in via the appropriately named argument "feed_dict"
    # v needs to be of shape=(5, 1) not just shape=(5,)
    # it's more like "real" matrix multiplication
    output = session.run(w, feed_dict={A: np.random.randn(5, 5), v: np.random.randn(5, 1)})

    # what's this output that is returned by the session? let's print it
    print(output, type(output))
    # luckily, the output type is just a numpy array. back to safety!


# TensorFlow variables are like Theano shared variables.
# But Theano variables are like TensorFlow placeholders.
# Are you confused yet?

# A tf variable can be initialized with a numpy array or a tf array
# or more correctly, anything that can be turned into a tf tensor
shape = (2, 2)
x = tf.Variable(tf.random_normal(shape))
t = tf.Variable(0) # a scalar

# you need to "initialize" the variables first
# here, x and t are global variables
init = tf.global_variables_initializer()

with tf.Session() as session:
    out = session.run(init) # and then "run" the init operation
    print(out) # it's just None

    print(x.eval()) # the initial value of x
    print(t.eval())


# let's now try to find the minimum of a simple cost function like we did in Theano
u = tf.Variable(20.0)
loss = u*u + u + 1.0

# One difference between Theano and TensorFlow is that you don't write the updates
# yourself in TensorFlow. You choose an optimizer that implements the algorithm you want.
# 0.3 is the learning rate. Documentation lists the params.
train_op = tf.train.GradientDescentOptimizer(0.3).minimize(loss)

# let's run a session again
init = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(init)

    # Strangely, while the weight update is automated, the loop itself is not!
    # So we'll just call train_op until convergence.
    # This is useful for us anyway since we want to track the loss function.
    for i in range(12):
        session.run(train_op)
        print("i = %d, loss = %.3f, u = %.3f" % (i, loss.eval(), u.eval()))

