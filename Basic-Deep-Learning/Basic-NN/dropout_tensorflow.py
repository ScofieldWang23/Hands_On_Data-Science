import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from util import get_normalized_data
from sklearn.utils import shuffle


class HiddenLayer(object):
    def __init__(self, M1, M2):
        self.M1 = M1
        self.M2 = M2
        W = np.random.randn(M1, M2) * np.sqrt(2.0 / M1)
        b = np.zeros(M2)
        self.W = tf.Variable(W.astype(np.float32))
        self.b = tf.Variable(b.astype(np.float32))
        self.params = [self.W, self.b]

    def forward(self, X):
        return tf.nn.relu(tf.matmul(X, self.W) + self.b)


class ANN(object):
    def __init__(self, hidden_layer_sizes, p_keep):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout_rates = p_keep # a list of probabilities

    def fit(self, X, Y, Xvalid, Yvalid, 
            lr=1e-4, mu=0.9, decay=0.9, 
            epochs=15, batch_sz=100, print_every=50):

        X = X.astype(np.float32)
        Y = Y.astype(np.int64)
        Xvalid = Xvalid.astype(np.float32)
        Yvalid = Yvalid.astype(np.int64)

        # initialize hidden layers
        N, D = X.shape
        K = len(set(Y)) # set(Y): get unique number of Y
        self.hidden_layers = []
        M1 = D
        for M2 in self.hidden_layer_sizes:
            h = HiddenLayer(M1, M2)
            self.hidden_layers.append(h)
            M1 = M2 # prepare for the next hidden layer

        W = np.random.randn(M1, K) * np.sqrt(2.0 / M1)
        b = np.zeros(K)
        self.W = tf.Variable(W.astype(np.float32))
        self.b = tf.Variable(b.astype(np.float32))

        # collect params for later use
        self.params = [self.W, self.b]
        for h in self.hidden_layers:
            self.params += h.params

        # set up tf functions and variables
        inputs = tf.placeholder(tf.float32, shape=(None, D), name='inputs')
        labels = tf.placeholder(tf.int64, shape=(None,), name='labels')
        logits = self.forward(inputs)
        # sparse_softmax_cross_entropy_with_logits allow us to 
        # directly pass labels instead of an indicator matrix of target
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits,
                labels=labels
            )
        )
        train_op = tf.train.RMSPropOptimizer(lr, decay=decay, momentum=mu).minimize(loss)
        # train_op = tf.train.MomentumOptimizer(lr, momentum=mu).minimize(loss)
        # train_op = tf.train.AdamOptimizer(lr).minimize(loss)
        prediction = self.predict(inputs)

        # validation loss will be calculated separately since nothing will be dropped
        test_logits = self.forward_test(inputs)
        test_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=test_logits,
                labels=labels
            )
        )

        n_batches = N // batch_sz
        Losses = []
        init = tf.global_variables_initializer()

        with tf.Session() as session:
            session.run(init)
            for i in range(epochs):
                print("epoch:", i, "n_batches:", n_batches)
                X, Y = shuffle(X, Y) # don't forget!
                for j in range(n_batches):
                    Xbatch = X[j * batch_sz:(j * batch_sz + batch_sz)]
                    Ybatch = Y[j * batch_sz:(j * batch_sz + batch_sz)]
                    # note inputs is Xbatch during training, Xvalid during validation and loss ploting
                    # ofc you can plot training loss, just replace Xvalid with X
                    session.run(train_op, feed_dict={inputs: Xbatch, labels: Ybatch})

                    if j % print_every == 0:
                        l = session.run(test_loss, feed_dict={inputs: Xvalid, labels: Yvalid})
                        p = session.run(prediction, feed_dict={inputs: Xvalid})
                        Losses.append(l)
                        e = error_rate(Yvalid, p)
                        print("i:", i, "j:", j, "nb:", n_batches, "loss:", l, "error rate:", e)
        
        plt.plot(Losses)
        plt.show()

    def forward(self, X):
        # tf.nn.dropout scales inputs by (1 / p_keep), which is known as "inverse dropout"
        # therefore, during test time, we don't have to scale anything
        Z = X
        Z = tf.nn.dropout(Z, self.dropout_rates[0])

        for h, p in zip(self.hidden_layers, self.dropout_rates[1:]):
            Z = h.forward(Z)
            Z = tf.nn.dropout(Z, p)

        return tf.matmul(Z, self.W) + self.b

    def forward_test(self, X):
        Z = X
        for h in self.hidden_layers:
            Z = h.forward(Z)

        return tf.matmul(Z, self.W) + self.b

    def predict(self, X):
        pY = self.forward_test(X)
        return tf.argmax(pY, 1)


def error_rate(p, t):
    return np.mean(p != t)


def relu(a):
    return a * (a > 0)


def main():
    # step 1: get the data and define all the usual variables
    Xtrain, Xtest, Ytrain, Ytest = get_normalized_data()

    ann = ANN([500, 300], [0.8, 0.5, 0.5]) # 1 hidden layer NN
    ann.fit(Xtrain, Ytrain, Xtest, Ytest)


if __name__ == '__main__':
    main()