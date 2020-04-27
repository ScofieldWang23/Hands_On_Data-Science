import os
import json
import numpy as np
from time import time
import matplotlib.pyplot as plt

from sklearn.utils import shuffle

import sys
sys.path.append(os.path.abspath('..'))

from util import get_wiki, find_analogies
from brown import get_text_with_word2idx_limit_vocab, get_text_with_word2idx

# using ALS (Alternating Least Squares), what's the least # files to get correct analogies?
# use this for word2vec training to make it faster
# first tried 20 files --> not enough
# how about 30 files --> some correct but still not enough
# 40 files --> half right but 50 is better

class Glove:
    def __init__(self, V, D, context_size):
        '''
        V: int, vocabulary size

        D: int, number of dimension

        '''
        self.V = V
        self.D = D
        self.context_size = context_size

    def train(self, sentences, cc_matrix=None, 
            learning_rate=1e-4, reg=0.1, xmax=100, 
            alpha=0.75, epochs=10, use_gd=False):
        '''
        sentences:

        cc_matrix: string, path to load the calculated cc_matrix

        reg:

        xmax:

        alpha: 

        use_gd: bool, default is False, indicate whether using gradient descent to train the model
                if False, use ALS() to train the model

        ''' 
        # build co-occurrence matrix
        # paper calls it X, so we will call it X, instead of calling the training data X
        # TODO: would it be better to use a sparse matrix to store X?

        t0 = time()
        V = self.V
        D = self.D

        # sentences is only used to create cc_matrix
        # we will not use sentences later
        if not os.path.exists(cc_matrix):
            # creating co-occurrence matrix X
            # here, co-occurrence includes words appearing within the context_size
            X = np.zeros((V, V))
            N = len(sentences)
            print("number of sentences to process:", N)
            it = 0 
            for sentence in sentences:
                it += 1
                if it % 10000 == 0:
                    print("processed", it, "/", N)

                n_word = len(sentence)
                for i in range(n_word):
                    # Don't confuse:
                    # i is not the word index!!!
                    # j is not the word index!!!
                    # i just points to which element of the sequence (sentence) we're looking at
                    wi = sentence[i]

                    start = max(0, i - self.context_size)
                    end = min(n_word, i + self.context_size)

                    # we can either choose only one side as context, or both
                    # here we are doing both

                    # make sure "start" and "end" tokens are part of some context
                    # otherwise their f(X) will be 0, which will appear in denominator in bias update)
                    # start token 0
                    if i - self.context_size < 0:
                        points = 1.0 / (i + 1) # ?
                        X[wi, 0] += points
                        X[0, wi] += points
                    # end token 1
                    if i + self.context_sz > n_word:
                        points = 1.0 / (n_word - i) # ?
                        X[wi, 1] += points
                        X[1, wi] += points

                    # left side
                    for j in range(start, i):
                        wj = sentence[j]
                        points = 1.0 / (i - j) # this is +ve
                        # every time wi, wj co-occurs, we plus the points(probs)
                        X[wi, wj] += points
                        X[wj, wi] += points

                    # right side
                    for j in range(i + 1, end):
                        wj = sentence[j]
                        points = 1.0 / (j - i) # this is +ve
                        X[wi, wj] += points
                        X[wj, wi] += points

            # save the cc matrix (file_name) because it takes forever to create
            np.save(cc_matrix, X)
        else:
            X = np.load(cc_matrix)

        print("max in cc matrix X is: ", X.max())

        # weighting
        fX = np.zeros((V, V))
        fX[X < xmax] = (X[X < xmax] / float(xmax)) ** alpha
        fX[X >= xmax] = 1
        print("max in f(X): ", fX.max())

        # target
        logX = np.log(X + 1) # +1 is important, logX is a matrix
        print("max in log(X): ", logX.max())
        print("time to build co-occurrence matrix: ", (time() - t0))

        # initialize weights
        W = np.random.randn(V, D) / np.sqrt(V + D) # V:fan_in, D:fan_out
        U = np.random.randn(V, D) / np.sqrt(V + D)
        b = np.zeros(V)
        c = np.zeros(V)
        mu = logX.mean() # this is optional


        losses = []
        sentence_indexes = range(len(sentences))
        for epoch in range(epochs):
            # make sure b and c are broadcasting in a right way!!!
            # b is column vector, c is row vector
            # (V,V) + (V,1) + (1,V) + sclar? - (V,V) -- numpy broadcast
            # delta is the common part used in the update, shape: (V,V)
            delta = W.dot(U.T) + b.reshape(V, 1) + c.reshape(1, V) + mu - logX
            loss = (fX * delta * delta).sum()
            losses.append(loss)
            print("epoch:", epoch, "loss:", loss)

            if use_gd: # gradient descent method
                # shape of both W and U are: (V, D)
                # update W
                # oldW = W.copy()
                for i in range(V):
                    W[i] -= learning_rate * (fX[i,:] * delta[i,:]).dot(U) # ((1,V) * (1,V)).dot(V,D)
                W -= learning_rate * reg * W # add regularization

                # update U
                for j in range(V):
                    U[j] -= learning_rate * (fX[:,j] * delta[:,j]).dot(W)
                U -= learning_rate * reg * U

                # update b
                for i in range(V):
                    b[i] -= learning_rate * fX[i,:].dot(delta[i,:])
                # b -= learning_rate * reg * b

                # update c, similar to b
                for j in range(V):
                    c[j] -= learning_rate * fX[:,j].dot(delta[:,j])
                # c -= learning_rate * reg * c


            else: # ALS method
                # update W
                for i in range(V):
                    matrix = reg * np.eye(D) + (fX[i, :] * U.T).dot(U)
                    vector = (fX[i, :] * (logX[i, :] - b[i] - c - mu)).dot(U)
                    W[i] = np.linalg.solve(matrix, vector) 

                # update U
                for j in range(V):
                    matrix = reg * np.eye(D) + (fX[:, j] * W.T).dot(W)
                    vector = (fX[:, j] * (logX[:, j] - b - c[j] - mu)).dot(W)
                    U[j] = np.linalg.solve(matrix, vector)

                # update b
                for i in range(V):
                    denominator = fX[i, :].sum() + reg
                    # assert(denominator > 0)
                    numerator = fX[i, :].dot(logX[i, :] - W[i].dot(U.T) - c - mu)
                    b[i] = numerator / denominator
                
                # update c
                for j in range(V):
                    denominator = fX[:, j].sum() + reg
                    numerator = fX[:, j].dot(logX[:, j] - W.dot(U[j]) - b  - mu)
                    c[j] = numerator / denominator
        
        # save W, U after training
        self.W = W
        self.U = U

        plt.plot(losses)
        plt.show()


    def save(self, save_dir, glove_file):
        '''

        '''
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        # function word_analogies expects a (V,D) matrx and a (D,V) matrix
        arrays = [self.W, self.U.T]
        # arrays = self.W
        # arrays = (self.W + self.U.T) / 2
        np.savez(os.path.join(save_dir, glove_file), *arrays)


def test_model(save_dir, glove_file, word2idx_file, use_brown=True, n_files=100):
    if use_brown:
        cc_matrix = "cc_matrix_brown.npy"
    else:
        cc_matrix = "cc_matrix_%s.npy" % n_files

    # hacky way of checking if we need to re-load the raw data or not
    # remember, only the co-occurrence matrix is needed for training!
    if os.path.exists(os.path.join(save_dir, cc_matrix)):
        with open(os.path.join(save_dir, word2idx_file)) as f:
            word2idx = json.load(f)
        sentences = [] # dummy - we won't actually use it

    else:# no saved cc_matrix and word2idx
        if use_brown:
            keep_words = set([
                'king', 'man', 'woman',
                'france', 'paris', 'london', 'rome', 'italy', 'britain', 'england',
                'french', 'english', 'japan', 'japanese', 'chinese', 'italian',
                'australia', 'australian', 'december', 'november', 'june',
                'january', 'february', 'march', 'april', 'may', 'july', 'august',
                'september', 'october',
            ])
            sentences, word2idx = get_text_with_word2idx_limit_vocab(n_vocab=5000, keep_words=keep_words)
        else:
            sentences, word2idx = get_wiki()
            # get_wikipedia_data(n_files=n_files, n_vocab=2000)
        
        # save the word2idx
        with open(os.path.join(save_dir, word2idx_file), 'w') as f:
            json.dump(word2idx, f)

    # training the Golve model
    Vocab_size = len(word2idx)
    model = Glove(Vocab_size, D=100, context_size=10)

    # alternating least squares method
    model.train(sentences, cc_matrix=cc_matrix, epochs=20)

    # # gradient descent method
    # model.train(
    #     sentences,
    #     cc_matrix=cc_matrix,
    #     learning_rate=5e-4,
    #     reg=0.1,
    #     epochs=500,
    #     use_gd=True,
    # )
    model.save(save_dir, glove_file)


if __name__ == '__main__':
    save_dir = os.path.join(os.getcwd(), 'we_model')
    glove_file = 'glove_model_50.npz'
    word2idx_file = 'glove_word2idx_50.json'
    # glove_file = 'glove_model_brown.npz'
    # word2idx = 'glove_word2idx_brown.json'

    test_model(save_dir, glove_file, word2idx_file, use_brown=False)
    
    # load back embeddings
    npz = np.load(glove_file)
    W1 = npz['arr_0']
    W2 = npz['arr_1']

    with open(word2idx_file) as f:
        word2idx = json.load(f)
        idx2word = {i:w for w,i in word2idx.items()}

    for concat in (True, False):
        print("** concat:", concat)

        if concat:
            We = np.hstack([W1, W2.T]) # We is (V, 2D)
        else:
            We = (W1 + W2.T) / 2 # We is (V, D)


        find_analogies('king', 'man', 'woman', We, word2idx, idx2word)
        find_analogies('france', 'paris', 'london', We, word2idx, idx2word)
        find_analogies('france', 'paris', 'rome', We, word2idx, idx2word)
        find_analogies('paris', 'france', 'italy', We, word2idx, idx2word)
        find_analogies('france', 'french', 'english', We, word2idx, idx2word)
        find_analogies('japan', 'japanese', 'chinese', We, word2idx, idx2word)
        find_analogies('japan', 'japanese', 'italian', We, word2idx, idx2word)
        find_analogies('japan', 'japanese', 'australian', We, word2idx, idx2word)
        find_analogies('december', 'november', 'june', We, word2idx, idx2word)