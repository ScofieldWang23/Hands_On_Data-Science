
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import sys
# sys.path.append(os.path.abspath('..'))

from sklearn.utils import shuffle
from utils import init_weight
from datetime import datetime
from sklearn.metrics import f1_score

from tensorflow.contrib.rnn import static_rnn as get_rnn_output
from tensorflow.contrib.rnn import BasicRNNCell, GRUCell

from utils import get_chunking_data

def flatten(l):
  # flatten a list of lists
  return [item for sublist in l for item in sublist]


# get the data
Xtrain, Ytrain, Xtest, Ytest, word2idx = get_chunking_data(split_sequences=True)
V = len(word2idx) + 2 # vocab size (+1 for unknown, +1 b/c start from 1)
K = len(set(flatten(Ytrain)) | set(flatten(Ytest))) + 1 # num classes

# training config
epochs = 10 # 20
learning_rate = 1e-2
mu = 0.99
batch_size = 32
hidden_layer_size = 10
embedding_dim = 10
sequence_length = max(len(x) for x in Xtrain + Xtest) # this is important! T = sequence_length


# pad sequences
Xtrain = tf.keras.preprocessing.sequence.pad_sequences(Xtrain, maxlen=sequence_length)
Ytrain = tf.keras.preprocessing.sequence.pad_sequences(Ytrain, maxlen=sequence_length)
Xtest  = tf.keras.preprocessing.sequence.pad_sequences(Xtest,  maxlen=sequence_length)
Ytest  = tf.keras.preprocessing.sequence.pad_sequences(Ytest,  maxlen=sequence_length)
print("Xtrain.shape:", Xtrain.shape)
print("Ytrain.shape:", Ytrain.shape)


# inputs
inputs = tf.placeholder(tf.int32, shape=(None, sequence_length)) # (N, T, 1)
targets = tf.placeholder(tf.int32, shape=(None, sequence_length)) # (N, T, 1)
num_samples = tf.shape(inputs)[0] # useful for later
# embedding
We = np.random.randn(V, embedding_dim).astype(np.float32) # (V, D)
# output layer
Wo = init_weight(hidden_layer_size, K).astype(np.float32) # (M, K)
bo = np.zeros(K).astype(np.float32) # (K,)

# make them tensorflow variables
tfWe = tf.Variable(We)
tfWo = tf.Variable(Wo)
tfbo = tf.Variable(bo)

# make the rnn unit
rnn_unit = GRUCell(num_units=hidden_layer_size, activation=tf.nn.relu)

# get the output
x = tf.nn.embedding_lookup(tfWe, inputs) #  (N, T, D)
# converts x from a tensor of shape (N, T, D)
# into a list of length T, where each element is a tensor of shape (N, D)
x = tf.unstack(x, sequence_length, 1) # ()

# get the rnn output
outputs, states = get_rnn_output(rnn_unit, x, dtype=tf.float32) # (T, N, M)

# outputs are now of size (T, N, M)
# so make it (N, T, M)
outputs = tf.transpose(outputs, (1, 0, 2)) # (N, T, M)
outputs = tf.reshape(outputs, (num_samples * sequence_length, hidden_layer_size)) # (NT, M)

# final dense layer
logits = tf.matmul(outputs, tfWo) + tfbo # (NT, K)
predictions = tf.argmax(logits, 1) # (NT, )
predict_op = tf.reshape(predictions, (num_samples, sequence_length)) # (N, T)
labels_flat = tf.reshape(targets, [-1]) # flattens shape into 1-D: (N, T, 1) --> (NT, )

loss_op = tf.reduce_mean(
  tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=logits,
    labels=labels_flat
  )
)

train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_op)

# init stuff
sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)

# training loop
losses = []
n_batches = len(Ytrain) // batch_size
for i in range(epochs):
  n_total = 0
  n_correct = 0

  t0 = datetime.now()
  Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
  loss = 0

  for j in range(n_batches):
    x = Xtrain[j * batch_size: (j+1) * batch_size]
    y = Ytrain[j * batch_size: (j+1) * batch_size]

    # get the loss, predictions, and perform a gradient descent step
    c, p, _ = sess.run(
      (loss_op, predict_op, train_op),
      feed_dict={inputs: x, targets: y})
    loss += c

    # calculate the accuracy
    for yi, pi in zip(y, p):
      # we don't care about the padded entries so ignore them!
      yii = yi[yi > 0]
      pii = pi[yi > 0]
      n_correct += np.sum(yii == pii)
      n_total += len(yii)

    # print stuff out periodically
    if j % 20 == 0: # 10
      sys.stdout.write(
        "j/N: %d/%d correct rate so far: %f, cost so far: %f\r" %
        (j, n_batches, float(n_correct)/n_total, loss)
      )
      sys.stdout.flush()

  # get test acc. too
  p = sess.run(predict_op, feed_dict={inputs: Xtest, targets: Ytest})
  n_test_correct = 0
  n_test_total = 0

  for yi, pi in zip(Ytest, p):
    yii = yi[yi > 0]
    pii = pi[yi > 0]
    n_test_correct += np.sum(yii == pii)
    n_test_total += len(yii)
  test_acc = float(n_test_correct) / n_test_total

  print(
      "i:", i, "cost:", "%.4f" % loss,
      "train acc:", "%.4f" % (float(n_correct)/n_total),
      "test acc:", "%.4f" % test_acc,
      "time for epoch:", (datetime.now() - t0)
  )
  losses.append(loss)

plt.plot(losses)
plt.show()


