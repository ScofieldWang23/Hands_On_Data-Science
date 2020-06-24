import os
import sys
import string

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid
from scipy.spatial.distance import cosine as cos_dist
from time import time
# from util import find_analogies

from sklearn.utils import shuffle
from sklearn.metrics.pairwise import pairwise_distances

from utils import get_wiki
# from brown import get_text_with_word2idx as get_brown
from brown import get_text_with_word2idx_limit_vocab as get_brown



class Word2Vec:
    def __init__(self):
        '''

        '''
        # get the data
        self.sentences, self.word2idx = get_wiki() # get_brown() takes much less time
        self.vocab_size = len(self.word2idx)
        self.idx2word = {i:w for w, i in self.word2idx.items()}


    def get_context(self, position, sentence):
        '''
        input:
        a sentence of the form: x x x x c c c pos c c c x x x x
        
        output:
        the context word indices: c c c c c c

        '''
        start = max(0, position - self.window_size)
        end  = min(len(sentence), position + self.window_size)

        context = []
        for ctx_pos, ctx_word_idx in enumerate(sentence[start:end], start=start):
            # don't include the input word itself as a target
            if ctx_pos != position:
                context.append(ctx_word_idx)
        return context 


    def get_negative_sampling_distribution(self):
        """
        Pn(w) = prob of word occuring
        we would like to sample the negative samples in such a way 
        that words occurring more often should be sampled more often

        """
        word_freq = np.ones(self.vocab_size) # word_id --> count, zeros()
        # word_count = sum(len(sentence) for sentence in self.sentences)
        for sentence in self.sentences:
            for word in sentence:
                word_freq[word] += 1

        # smooth it, give rare words larger chance of being selected
        p_neg = word_freq ** 0.75
        # normalize it, don't forget!
        p_neg = p_neg / p_neg.sum()

        assert(np.all(p_neg > 0))
        return p_neg # a vector of probability, length: vocab_size


    def sgd(self, word_id, targets, y):
        '''
        W[word_id] shape: D, just for a single center word, so it's D!
        V[:,targets] shape: D x n, n: number of context words, e.g. 5
        activation shape: n

        W: (Vocab_size, D), e.g. (8000, 100)
        V: (D, Vocab_size), e.g. (100, 8000)

        word_id: int, index of one word

        targets: list, indices of context words

        y: int, 1 or 0 --> represents positive or negative word/sample

        '''
        # print("word_id:", word_id, "targets:", targets)
        activation = self.W[word_id].dot(self.V[:,targets]) # also known as logits
        prob = sigmoid(activation) # shape: (n,) n context words

        # update gradients of relevant part of V and W
        gV = np.outer(self.W[word_id], prob - y) # outer product: D x n, label is just a int scalar values
        gW = np.sum((prob - y) * self.V[:,targets], axis=1) # sum(n * (D,n), axis=1), apply sum on each row --> D

        self.V[:,targets] -= self.lr_rate * gV # D x n
        self.W[word_id] -= self.lr_rate * gW # D

        # return loss (binary cross entropy)
        # add 1e-10 to avoid 0
        loss = -(y * np.log(prob + 1e-10) + (1 - y) * np.log(1 - prob + 1e-10))
        return loss.sum()
    
    
    def train(self, save_path, window_size=5, lr_rate=0.025, 
              num_negatives=5, epochs=20, hidden_size=50, 
              embedding_option='first', show_loss_plot=True):
        '''
        embedding_option: string, 'first' means use the first input-to-hidden weight matrix(vocab_size, D) as final embeddings
                                  'average' means use the average of W and V.T as final embeddings

        '''
        self.window_size = window_size
        self.lr_rate = lr_rate
        self.final_lr_rate = 0.0001
        # number of negative samples to draw per input word
        self.num_negatives = num_negatives
        self.epochs = epochs
        self.D = hidden_size # word embedding sizes
        # hack the learning rate decay
        self.lr_rate_delta = (lr_rate - self.final_lr_rate) / epochs

        # params initialization
        self.W = np.random.randn(self.vocab_size, self.D) # input-to-hidden
        self.V = np.random.randn(self.D, self.vocab_size) # hidden-to-output

        # distribution for drawing negative samples
        p_neg = self.get_negative_sampling_distribution()

        # save the losses to plot them per iteration
        self.losses = []

        # number of total words in corpus
        total_words = sum(len(sentence) for sentence in self.sentences)
        print("total number of words in corpus:", total_words)

        # for subsampling each sentence, drop frequent words while keep rare words
        threshold = 1e-5
        # p_drop is propotional to p_neg
        p_drop = 1 - np.sqrt(threshold / p_neg)

        # train the model, very slow if using wiki data
        for epoch in range(self.epochs):
            # shuffle sentences so we don't always see sentences in the same order
            np.random.shuffle(self.sentences)

            # accumulate the cost
            loss = 0
            counter = 0
            t0 = time()
            for sentence in self.sentences:
                # keep only certain words based on p_neg == randomly drop some words in the sentence
                # e.g. the bigger p_drop[w] is, the smaller (1 - p_drop[w]) would be
                # the more likely w would be dropped
                sentence = [w for w in sentence \
                                if np.random.random() < (1 - p_drop[w])
                ]
                if len(sentence) < 2: # less than 2 words meaning no context word!
                    continue

                # randomly order words so we don't always see
                # samples(center word) in the same order
                # ofc, this is optional
                randomly_ordered_positions = np.random.choice(
                    len(sentence),
                    size=len(sentence), #np.random.randint(1, len(sentence) + 1),
                    replace=False,
                )
                # think of pos as the index of a list
                for pos in randomly_ordered_positions:
                    # get the center word (word_id)
                    word = sentence[pos]

                    # get the positive context words & negative samples
                    context_words = self.get_context(pos, sentence)
                    neg_word = np.random.choice(self.vocab_size, p=p_neg)
                    # an array of context words id
                    targets = np.array(context_words) 

                    # do one iteration of stochastic gradient descent
                    loss += self.sgd(word, targets, 1)
                    loss += self.sgd(neg_word, targets, 0)

                # finish one training of a sentence
                counter += 1
                if counter % 100 == 0:
                    sys.stdout.write("processed {} / {}\r".format(counter, len(self.sentences)))
                    sys.stdout.flush()

            # print('p_neg', p_neg)
            delta_t = np.round((time() - t0), 2)
            print('epoch {} finished, loss: {}, time consumed: {}s'.format(epoch, loss, delta_t))

            # save the loss
            self.losses.append(loss)

            # update the learning rate
            self.lr_rate -= self.lr_rate_delta

        # get the fianl word emebddings!
        if embedding_option == 'first':
            self.wv = self.W
        else:
            self.wv = (self.W + self.V.T) / 2

        if show_loss_plot:
            plt.plot(self.losses)
            plt.show()

        # save the model
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        
        if not os.path.exists(os.path.join(save_path, 'word2idx.json')):
            with open('{}/word2idx.json'.format(save_path), 'w') as f:
                json.dump(self.word2idx, f)

        np.savez('{}/weights.npz'.format(save_path), self.wv)

        # return self.word2idx, self.W, self.V


    def analogy(self, pos1, neg1, pos2, neg2):
        '''

        '''
        N, D = self.W.shape
        assert(N == self.vocab_size)

        # don't actually use pos2 in calculation, just print what's expected
        print("testing: %s - %s = %s - %s" % (pos1, neg1, pos2, neg2))
        for w in (pos1, neg1, pos2, neg2):
            if w not in self.word2idx:
                print("Sorry, %s not in pre-trained word2idx" % w)
                return

        p1_w2v = self.wv[self.word2idx[pos1]]
        n1_w2v = self.wv[self.word2idx[neg1]]
        p2_w2v = self.wv[self.word2idx[pos2]]
        n2_w2v = self.wv[self.word2idx[neg2]]

        w2v = p1_w2v - n1_w2v + n2_w2v # (D,)

        distances = pairwise_distances(w2v.reshape(1, D), self.W, metric='cosine').reshape(N)
        idx = distances.argsort()[:10] # smaller distance, more similar

        # pick one that's not p1, n1, or n2
        best_idx = -1
        idx_not_use = [self.word2idx[w] for w in (pos1, neg1, neg2)]
        for i in idx:
            if i not in idx_not_use:
                best_idx = i
                break

        print("analogy results: %s - %s = %s - %s" % (pos1, neg1, self.idx2word[best_idx], neg2))
        print("closest top 10:")
        for i in idx:
            print(self.idx2word[i], distances[i])

        print("cosine distance to %s:" % pos2, cos_dist(p2_w2v, w2v))


def load_model(dir):
    '''

    '''
    with open('%s/word2idx.json' % dir) as f:
        word2idx = json.load(f)

    npz = np.load('%s/weights.npz' % dir)
    wv = npz['arr_0']
    return word2idx, wv


def test_model(save_path):
    '''
    there are multiple ways to get the "final" word embedding
        1. We = (W + V.T) / 2
        2. We = W

    '''
    w2v_model = Word2Vec()
    w2v_model.train(epochs=20, save_path=save_path)
    
    print("**********")
    w2v_model.analogy('king', 'man', 'queen', 'woman')
       

if __name__ == "__main__":
    # idx_text, fword2idx = get_brown()
    path = os.path.join(os.getcwd(), 'w2v_model')
    test_model(save_path=path)
    