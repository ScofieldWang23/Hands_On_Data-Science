import numpy as np
import os
import sys
import string
import operator

from sklearn.metrics.pairwise import pairwise_distances
from glob import glob


def init_weight(Mi, Mo):
    return np.random.randn(Mi, Mo) / np.sqrt(Mi + Mo)


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

        
# unfortunately these work different ways
def remove_punctuation_2(s):
    return s.translate(None, string.punctuation)

def remove_punctuation_3(s):
    return s.translate(str.maketrans('','',string.punctuation))

if sys.version.startswith('2'):
    remove_punctuation = remove_punctuation_2
else:
    remove_punctuation = remove_punctuation_3


def get_wiki(Vocab=20000):
    '''
    
    '''
    V = Vocab # top frequent words
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

    top_words = ['START', 'END'] + [w for w, count in all_word_counts[ : V]] + ['<UNK>']
    # top_words = [w for w, count in all_word_counts[ : V-1]] + ['<UNK>']
    word2idx = {w:i for i, w in enumerate(top_words)}
    unk = word2idx['<UNK>']

    # data sanity check
    assert('START' in word2idx)
    assert('END' in word2idx)
    assert('king' in word2idx)
    assert('queen' in word2idx)
    assert('man' in word2idx)
    assert('woman' in word2idx)

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


def get_wikipedia_data(n_files, n_vocab, by_paragraph=False):
    prefix = 'data/enwiki_corpus/'

    if not os.path.exists(prefix):
        print("Are you sure you've downloaded, converted, and placed the Wikipedia data into the proper folder?")
        print("I'm looking for a folder called large_files, adjacent to the class folder, but it does not exist.")
        print("Please download the data from https://dumps.wikimedia.org/")
        print("Quitting...")
        exit()

    input_files = [f for f in os.listdir(prefix) if f.startswith('enwiki') and f.endswith('txt')]

    if len(input_files) == 0:
        print("Looks like you don't have any data files, or they're in the wrong location.")
        print("Please download the data from https://dumps.wikimedia.org/")
        print("Quitting...")
        exit()

    # return variables
    sentences = []
    word2idx = {'START': 0, 'END': 1}
    idx2word = ['START', 'END']
    current_idx = 2
    # set to infinity, we need to sort later
    word_idx_count = {0: float('inf'), 1: float('inf')}

    if n_files is not None:
        input_files = input_files[:n_files]

    for f in input_files:
        print("reading:", f)
        for line in open(prefix + f):
            line = line.strip()
            # don't count headers, structured data, lists, etc...
            if line and line[0] not in ('[', '*', '-', '|', '=', '{', '}'):
                if by_paragraph:
                    sentence_lines = [line]
                else:
                    sentence_lines = line.split('. ')
                for sentence in sentence_lines:
                    # tokens = my_tokenizer(sentence)
                    tokens = remove_punctuation(line).lower().split()
                    for t in tokens:
                        if t not in word2idx:
                            word2idx[t] = current_idx
                            idx2word.append(t)
                            current_idx += 1
                        idx = word2idx[t]
                        word_idx_count[idx] = word_idx_count.get(idx, 0) + 1
                    # sentence_by_idx would be very convenient in use! 
                    sentence_by_idx = [word2idx[t] for t in tokens] 
                    sentences.append(sentence_by_idx)
    
    # restrict vocab size, sort in descending word count order
    sorted_word_idx_count = sorted(word_idx_count.items(), key=operator.itemgetter(1), reverse=True)
    word2idx_small = {}
    new_idx = 0
    idx_new_idx_map = {}
    for idx, count in sorted_word_idx_count[:n_vocab]:
        word = idx2word[idx]
        print(word, count) # for debug
        word2idx_small[word] = new_idx
        idx_new_idx_map[idx] = new_idx
        new_idx += 1
    # let 'unknown' be the last token
    word2idx_small['UNKNOWN'] = new_idx 
    unknown = new_idx
    # data sanity check
    assert('START' in word2idx_small)
    assert('END' in word2idx_small)
    assert('king' in word2idx_small)
    assert('queen' in word2idx_small)
    assert('man' in word2idx_small)
    assert('woman' in word2idx_small)

    # map old idx to new idx, each sentence is represented by a idx
    sentences_small = []
    for sentence in sentences:
        if len(sentence) > 1:
            new_sentence = [idx_new_idx_map[idx] if idx in idx_new_idx_map else unknown for idx in sentence]
            sentences_small.append(new_sentence)

    return sentences_small, word2idx_small
    


def get_chunking_data(split_sequences=False, start_index=1):
  if not os.path.exists('data/chunking'):
    print("Please create a folder in your local directory called 'chunking'")
    print("train.txt and test.txt should be stored in there.")
    print("Please check the comments to get the download link.")
    exit()
  elif not os.path.exists('data/chunking/train.txt'):
    print("train.txt is not in data/chunking/train.txt")
    print("Please check the comments to get the download link.")
    exit()
  elif not os.path.exists('data/chunking/test.txt'):
    print("test.txt is not in data/chunking/test.txt")
    print("Please check the comments to get the download link.")
    exit()

  word2idx = {}
  tag2idx = {}
  # idx starts from 1, because 0 is used for padding in TF
  word_idx = start_index
  tag_idx = start_index
  Xtrain = []
  Ytrain = []
  currentX = [] # a list of word index
  currentY = []

  for line in open('data/chunking/train.txt'):
    line = line.rstrip()
    if line:
      r = line.split()
      word, tag, _ = r
      if word not in word2idx:
        word2idx[word] = word_idx
        word_idx += 1
      currentX.append(word2idx[word])
      
      if tag not in tag2idx:
        tag2idx[tag] = tag_idx
        tag_idx += 1
      currentY.append(tag2idx[tag])

    elif split_sequences:
      Xtrain.append(currentX)
      Ytrain.append(currentY)
      currentX = []
      currentY = []

  if not split_sequences:
    Xtrain = currentX
    Ytrain = currentY

  # load and score test data
  Xtest = []
  Ytest = []
  currentX = []
  currentY = []
  for line in open('data/chunking/test.txt'):
    line = line.rstrip()
    if line:
      r = line.split()
      word, tag, _ = r
      if word in word2idx:
        currentX.append(word2idx[word])
      else:
        currentX.append(word_idx) # use this as unknown
      currentY.append(tag2idx[tag])

    elif split_sequences:
      Xtest.append(currentX)
      Ytest.append(currentY)
      currentX = []
      currentY = []

  if not split_sequences:
    Xtest = currentX
    Ytest = currentY

  return Xtrain, Ytrain, Xtest, Ytest, word2idx
