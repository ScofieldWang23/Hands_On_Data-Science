# 2-hidden layer NN in TensorFlow
# This code is not optimized for speed.
# It's just to get something working, using the principles we know.

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from util import get_normalized_data, y2indicator


def error_rate(p, t):
    return np.mean(p != t)


def main():
    # step 1: get the data and define all the usual variables
    Xtrain, Xtest, Ytrain, Ytest = get_normalized_data()

    max_iter = 15
    print_period = 50

    lr = 0.00004
    reg = 0.01

    Ytrain_ind = y2indicator(Ytrain)
    Ytest_ind = y2indicator(Ytest)

    N, D = Xtrain.shape
    batch_sz = 500
    n_batches = N // batch_sz

    # add an extra layer just for fun(now, we have 2 hidden layers)
    M1 = 300
    M2 = 100
    K = 10
    W1_init = np.random.randn(D, M1) / np.sqrt(D)
    b1_init = np.zeros(M1)
    W2_init = np.random.randn(M1, M2) / np.sqrt(M1)
    b2_init = np.zeros(M2)
    W3_init = np.random.randn(M2, K) / np.sqrt(M2)
    b3_init = np.zeros(K)

    # define variables and expressions
    # Note: shape can be (None, D), (None, K)
    X = tf.placeholder(tf.float32, shape=(None, D), name='X')
    T = tf.placeholder(tf.float32, shape=(None, K), name='T')

    W1 = tf.Variable(W1_init.astype(np.float32))
    b1 = tf.Variable(b1_init.astype(np.float32))
    W2 = tf.Variable(W2_init.astype(np.float32))
    b2 = tf.Variable(b2_init.astype(np.float32))
    W3 = tf.Variable(W3_init.astype(np.float32))
    b3 = tf.Variable(b3_init.astype(np.float32))

    # define the model
    # Z is the output of activation function
    Z1 = tf.nn.relu( tf.matmul(X, W1) + b1 )
    Z2 = tf.nn.relu( tf.matmul(Z1, W2) + b2 )
    # here Yish is not Y, cause we havn't done softmax
    # remember, the loss function does the softmaxing! 
    Yish = tf.matmul(Z2, W3) + b3 

    # softmax_cross_entropy_with_logits take in the "logits", which is Yish
    # if you wanted to know the actual output of the neural net,
    # you could pass "Yish" into tf.nn.softmax(logits)
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Yish, labels=T))

    # we choose the optimizer instead of implementing the algorithm by ourselves
    # let's go with RMSprop, since we just learned about it.
    # Note: it includes momentum!
    train_op = tf.train.RMSPropOptimizer(lr, decay=0.99, momentum=0.9).minimize(loss)
    # we'll use this to calculate the error rate
    predict_op = tf.argmax(Yish, 1)

    Losses = []
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init) # don't forget!

        for i in range(max_iter): # epoch, actually we should shuffle data in each epoch
            for j in range(n_batches):
                Xbatch = Xtrain[j * batch_sz: (j * batch_sz + batch_sz),]
                Ybatch = Ytrain_ind[j * batch_sz:( j * batch_sz + batch_sz),]

                session.run(train_op, feed_dict={X: Xbatch, T: Ybatch})
                if j % print_period == 0:
                    test_cost = session.run(loss, feed_dict={X: Xtest, T: Ytest_ind})
                    prediction = session.run(predict_op, feed_dict={X: Xtest})
                    err = error_rate(prediction, Ytest)
                    print("Cost / err at iteration i=%d, j=%d: %.3f / %.3f" % (i, j, test_cost, err))
                    Losses.append(test_cost)
 
    plt.plot(Losses)
    plt.show()
    # increase max_iter and notice how the test cost starts to increase.
    # are we overfitting by adding that extra layer?
    # how would you add regularization to this model?


if __name__ == '__main__':
    main()
