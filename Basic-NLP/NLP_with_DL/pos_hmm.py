
import numpy as np
import matplotlib.pyplot as plt

import os
import sys
# sys.path.append(os.path.abspath('..'))


from sklearn.utils import shuffle
from datetime import datetime
from sklearn.metrics import f1_score

from utils import get_chunking_data
from hmmd_scaled import HMM


def accuracy(T, Y):
    # inputs are lists of lists
    n_correct = 0
    n_total = 0
    for t, y in zip(T, Y):
        n_correct += np.sum(t == y)
        n_total += len(y)
    return float(n_correct) / n_total


def total_f1_score(T, Y):
    # inputs are lists of lists
    T = np.concatenate(T) # concat along the row: a list of lists --> 2D np.array
    Y = np.concatenate(Y)
    return f1_score(T, Y, average=None).mean()


def flatten(l):
    return [item for sublist in l for item in sublist]


def main(smoothing=1e-1):
    # X = words, Y = POS tags
    Xtrain, Ytrain, Xtest, Ytest, word2idx = get_chunking_data(split_sequences=True, start_index=0)
    V = len(word2idx) + 1

    # find hidden state transition matrix and pi
    # M = max(max(y) for y in Ytrain) + 1
    M = len(set(flatten(Ytrain))) + 1 # | set(flatten(Ytest))

    # find the transition matrix
    A = np.ones((M, M)) * smoothing # add-one smoothing
    pi = np.zeros(M)
    for y in Ytrain:
        pi[y[0]] += 1 
        for i in range(len(y) - 1):
            A[y[i], y[i+1]] += 1
    # turn it into a probability matrix
    A /= A.sum(axis=1, keepdims=True)
    pi /= pi.sum()

    # find the observation matrix
    B = np.ones((M, V))*smoothing # add-one smoothing
    for x, y in zip(Xtrain, Ytrain):
        for xi, yi in zip(x, y):
            B[yi, xi] += 1 # Be cautious: yi(hidden state) --> xi(observation)
    B /= B.sum(axis=1, keepdims=True)

    hmm = HMM(M)
    hmm.pi = pi
    hmm.A = A
    hmm.B = B

    # get predictions, just use the two matrices created based on counting!
    Ptrain = []
    for x in Xtrain:
        p = hmm.get_state_sequence(x)
        Ptrain.append(p)

    Ptest = []
    for x in Xtest:
        p = hmm.get_state_sequence(x)
        Ptest.append(p)

    # print results
    print("train accuracy:", accuracy(Ytrain, Ptrain))
    print("test accuracy:", accuracy(Ytest, Ptest))
    print("train f1:", total_f1_score(Ytrain, Ptrain))
    print("test f1:", total_f1_score(Ytest, Ptest))

if __name__ == '__main__':
    main()
