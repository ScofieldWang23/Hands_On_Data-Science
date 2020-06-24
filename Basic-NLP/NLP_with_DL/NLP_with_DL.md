
<!-- TOC -->

- [Description](#description)

<!-- /TOC -->


## Description
This folder is mainly for holding notebooks and .py files for learning NLP with Deep Learning.

This folder will cover **4 new architectures** used in NLP.

**(1) To begin with, you will learn how `word2vec` works, from theory to implementation**.

`Word2vec` is interesting because it magically maps words to a vector space where you can **find analogies**, like:

- king - man = queen - woman
- France - Paris = England - London
- December - Novemeber = July - June

For those beginners who find algorithms tough and just want to use a library, we will demonstrate the use of the `Gensim` library to obtain **pre-trained word vectors**, compute similarities and analogies, and apply those word vectors to build text classifiers. You will also learn how to build a word2vec model by using the `Spacy` library

**(2) Then we will focus on the `GloVe` method**, which also finds word vectors, but uses a technique called **matrix factorization**, which is a popular algorithm for recommender systems.

Amazingly, the word vectors produced by GLoVe are **just as good as** the ones produced by word2vec, and it’s **much easier** to train.

We will also look at some **classical NLP problems**, like **parts-of-speech tagging** and **named entity recognition**, and **(3) use `recurrent neural networks`** to solve them. You’ll see that any problem can be solved using neural networks, but you’ll also learn **the dangers of having too much complexity**.

**(4) Lastly, you’ll learn about `recursive neural networks`**, which finally help us solve the problem of **negation in sentiment analysis**. Recursive neural networks exploit the fact that sentences have a tree structure, and we can finally get away from naively using bag-of-words.

<br>

**What you’ll learn**
- Understand and implement `word2vec`
- Understand the `CBOW` method in `word2vec`
- Understand the `skip-gram` method in `word2vec`
- Understand the `negative sampling optimization` in `word2vec`
- Understand and implement `GloVe` using `gradient descent` and `alternating least squares`
- Use `recurrent neural networks` for `parts-of-speech tagging`
- Use `recurrent neural networks` for `named entity recognition`
- Understand and implement `recursive neural networks` for sentiment analysis
- Use `Gensim` to obtain pretrained word vectors and compute similarities and analogies

