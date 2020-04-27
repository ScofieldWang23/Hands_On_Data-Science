from nltk.corpus import brown
import operator

KEEP_WORDS = set([
  'king', 'man', 'queen', 'woman',
  'italy', 'rome', 'france', 'paris',
  'london', 'britain', 'england',
])


def get_text():
  # returns 57340 of the Brown corpus
  # each sentence is represented as a list of stokens
  return brown.sents()


def get_text_with_word2idx():
  '''
  return: - a list of sentences represented by word_id
          - a words dictionary -- word2idx

  '''
  text = get_text() # a list of list
  idx_text = [] # 

  i = 2
  # don't forget adding 'START' and 'END' tokenss
  word2idx = {'START': 0, 'END': 1}
  for sentence in text:
    idx_sentence = []
    for token in sentence:
      token = token.lower()
      if token not in word2idx:
        word2idx[token] = i
        i += 1

      idx_sentence.append(word2idx[token])
    idx_text.append(idx_sentence)

  print("Vocab size:", i)
  return idx_text, word2idx


def get_text_with_word2idx_limit_vocab(n_vocab=2000, keep_words=KEEP_WORDS):
  '''
  return: - a list of sampled sentences represented by word_id
          - a sampled words dictionary -- word2idx

  '''
  text = get_text()
  idx_text = []

  i = 2
  word2idx = {'START': 0, 'END': 1}
  idx2word = ['START', 'END']

  # keep the 'START' and 'END', set their count to infinity
  word_idx_count = {
    0: float('inf'),
    1: float('inf'),
  }

  for sentence in text:
    idx_sentence = []
    for token in sentence:
      token = token.lower()
      if token not in word2idx:
        idx2word.append(token)
        word2idx[token] = i
        i += 1

      # keep track of counts for later sorting
      idx = word2idx[token]
      word_idx_count[idx] = word_idx_count.get(idx, 0) + 1

      idx_sentence.append(idx)
    idx_text.append(idx_sentence)


  # restrict vocab size
  # set all the words I want to keep to infinity
  # so that they are included when I pick the most
  # common words
  for word in keep_words:
    word_idx_count[word2idx[word]] = float('inf')

  sorted_word_idx_count = sorted(word_idx_count.items(), key=operator.itemgetter(1), reverse=True)
  word2idx_small = {}
  new_idx = 0
  idx_new_idx_map = {}
  # only keep top n_vocab words
  for idx, count in sorted_word_idx_count[:n_vocab]:
    word = idx2word[idx]
    # print(word, count)
    word2idx_small[word] = new_idx
    idx_new_idx_map[idx] = new_idx
    new_idx += 1
  # Don't forget: let 'unknown' be the last token !!!
  word2idx_small['UNKNOWN'] = new_idx 
  unknown = new_idx

  assert('START' in word2idx_small)
  assert('END' in word2idx_small)
  for word in keep_words:
    assert(word in word2idx_small)

  # map old idx to new idx
  idx_text_small = []
  for sentence in idx_text:
    if len(sentence) > 1:
      new_sentence = [idx_new_idx_map[idx] if idx in idx_new_idx_map else unknown for idx in sentence]
      idx_text_small.append(new_sentence)

  return idx_text_small, word2idx_small


if __name__ == "__main__":
  ###### Test ######
  idx_text, word2idx = get_text_with_word2idx_limit_vocab()
  print('Total number of words in corpus'.format(len(word2idx)))
  print('word2idx representation of the first sentence in the text'.format(idx_text[0]))
