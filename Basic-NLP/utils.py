import os
import sys
import string
import operator

from sklearn.metrics.pairwise import pairwise_distances
from glob import glob


# unfortunately these work different ways
def remove_punctuation_2(s):
    return s.translate(None, string.punctuation)

def remove_punctuation_3(s):
    return s.translate(str.maketrans('','',string.punctuation))

if sys.version.startswith('2'):
    remove_punctuation = remove_punctuation_2
else:
    remove_punctuation = remove_punctuation_3


def get_wiki():
    '''
    
    '''
    V = 20000 # top frequent words
    files = glob('data/enwiki_corpus/enwiki*.txt')
    all_word_counts = {}
    for f in files:
        for line in open(f):
            if line and line[0] not in '[*-|=\{\}':
                s = remove_punctuation(line).lower().split()
                if len(s) > 1:
                    for word in s:
                        all_word_counts[word] = all_word_counts.get(word, 0) + 1
                        # if word not in all_word_counts:
                        #     all_word_counts[word] = 0
                        #     all_word_counts[word] += 1
    print("finished counting")

    V = min(V, len(all_word_counts))
    all_word_counts = sorted(all_word_counts.items(), key=lambda x: x[1], reverse=True)
    # all_word_counts = sorted(all_word_counts.items(), key=operator.itemgetter(1), reverse=True)

    top_words = [w for w, count in all_word_counts[ : V-1]] + ['<UNK>']
    word2idx = {w:i for i, w in enumerate(top_words)}
    unk = word2idx['<UNK>']

    sents = []
    for f in files:
        for line in open(f):
            if line and line[0] not in '[*-|=\{\}':
                s = remove_punctuation(line).lower().split()
                if len(s) > 1:
                    # if a word is not nearby another word, there won't be any context!
                    # and hence nothing to train!
                    sent = [word2idx[w] if w in word2idx else unk for w in s] # bow of sent
                    sents.append(sent)

    return sents, word2idx


# fast version
def find_analogies(w1, w2, w3, We, word2idx, idx2word):
    V, D = We.shape

    king = We[word2idx[w1]]
    man = We[word2idx[w2]]
    woman = We[word2idx[w3]]
    v0 = king - man + woman

    for dist in ('euclidean', 'cosine'):
        distances = pairwise_distances(v0.reshape(1, D), We, metric=dist).reshape(V)
        # idx = distances.argmin()
        # best_word = idx2word[idx]
        idx = distances.argsort()[:4]
        best_idx = -1
        keep_out = [word2idx[w] for w in (w1, w2, w3)]
        for i in idx:
            if i not in keep_out:
                best_idx = i
                break
        best_word = idx2word[best_idx]


        print("closest match by", dist, "distance:", best_word)
        print(w1, "-", w2, "=", best_word, "-", w3)