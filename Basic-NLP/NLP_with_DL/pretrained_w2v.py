
from gensim.models import KeyedVectors

# warning: takes quite awhile
# https://code.google.com/archive/p/word2vec/
# 3 million words and phrases, by comparison Glove vector only use 400,000 words
# During most of time, in real application, we only use words about 20,000
# D = 300
word_vectors = KeyedVectors.load_word2vec_format(
  './data/GoogleNews-vectors-negative300.bin',
  binary=True
)


# convenience
# result looks like:
# [('athens', 0.6001024842262268),
#  ('albert', 0.5729557275772095),
#  ('holmes', 0.569324254989624),
#  ('donnie', 0.5690680742263794),
#  ('italy', 0.5673537254333496),
#  ('toni', 0.5666348338127136),
#  ('spain', 0.5661854147911072),
#  ('jh', 0.5661597847938538),
#  ('pablo', 0.5631559491157532),
#  ('malta', 0.5620371103286743)]

def find_analogies(w1, w2, w3):
  r = word_vectors.most_similar(positive=[w1, w3], negative=[w2])
  print("%s - %s = %s - %s" % (w1, w2, r[0][0], w3))

def nearest_neighbors(w):
  r = word_vectors.most_similar(positive=[w])
  print("neighbors of: %s" % w)
  for word, score in r:
    print("\t%s" % word)


find_analogies('king', 'man', 'woman')
find_analogies('france', 'paris', 'london')
find_analogies('france', 'paris', 'rome')
find_analogies('paris', 'france', 'italy')
find_analogies('france', 'french', 'english')
find_analogies('japan', 'japanese', 'chinese')
find_analogies('japan', 'japanese', 'italian')
find_analogies('japan', 'japanese', 'australian')
find_analogies('december', 'november', 'june')
find_analogies('miami', 'florida', 'texas')
find_analogies('einstein', 'scientist', 'painter')
find_analogies('china', 'rice', 'bread')
find_analogies('man', 'woman', 'she')
find_analogies('man', 'woman', 'aunt')
find_analogies('man', 'woman', 'sister')
find_analogies('man', 'woman', 'wife')
find_analogies('man', 'woman', 'actress')
find_analogies('man', 'woman', 'mother')
find_analogies('heir', 'heiress', 'princess')
find_analogies('nephew', 'niece', 'aunt')
find_analogies('france', 'paris', 'tokyo')
find_analogies('france', 'paris', 'beijing')
find_analogies('february', 'january', 'november')
find_analogies('france', 'paris', 'rome')
find_analogies('paris', 'france', 'italy')

nearest_neighbors('king')
nearest_neighbors('france')
nearest_neighbors('japan')
nearest_neighbors('einstein')
nearest_neighbors('woman')
nearest_neighbors('nephew')
nearest_neighbors('february')
nearest_neighbors('rome')