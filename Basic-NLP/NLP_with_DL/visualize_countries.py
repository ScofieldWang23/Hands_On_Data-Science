import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_tsne(path_dir = os.path.join(os.getcwd(), 'Basic-NLP/NLP_with_DL/w2v_model'), 
        we_file='glove_model_wiki.npz', 
        w2i_file='glove_word2idx_wiki.json'):

    words = ['japan', 'japanese', 'england', 'english', 'australia', 'australian', 'china', 'chinese', \
            'italy', 'italian', 'french', 'france', 'spain', 'spanish']

    with open(os.path.join(path_dir, w2i_file)) as f:
        word2idx = json.load(f)
        print('successfully loaded the word2idx!')

    npz = np.load(os.path.join(path_dir, we_file))
    W = npz['arr_0']
    V = npz['arr_1']
    We = (W + V.T) / 2

    idx = [word2idx[w] for w in words]
    # We = We[idx]

    tsne = TSNE()
    Z = tsne.fit_transform(We)
    # just show selected words
    Z = Z[idx]
    plt.scatter(Z[:,0], Z[:,1])
    for i in range(len(words)):
        plt.annotate(s=words[i], xy=(Z[i,0], Z[i,1]))
    plt.show()


if __name__ == '__main__':
    plot_tsne()
