from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import sys
import logging
import pickle
import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM
from keras.preprocessing.sequence import pad_sequences
from keras_contrib.layers import CRF

EPOCHS = 10
EMBEDDING_DIM = 200
BiLSTM_HIDDEN_DIM = 200

error_dict = {
    'R': 1,
    'M': 2,
    'S': 3,
    'W': 4
}

def get_idx_from_sent(sent, word_idx_map):
    x = []

    for word in sent:
        if word in word_idx_map:
            x.append(word_idx_map[word])
        else:
            x.append(1)
    return x

def make_idx_data(revs, word_idx_map, maxlen=60):
    X_train, X_test, y_train = [], [], []
    for rev in revs:
        sent = get_idx_from_sent(rev['text'], word_idx_map)

        if rev['option'] == 'train':
            y = rev['label']

            X_train.append(sent)
            y_train.append(y)

        elif rev['option'] == 'test':
            X_test.append(sent)

    X_train = sequence.pad_sequences(np.array(X_train), maxlen=maxlen)
    X_test = sequence.pad_sequences(np.array(X_test), maxlen=maxlen)
    # X_valid = sequence.pad_sequences(np.array(X_valid), maxlen=maxlen)
    y_train = sequence.pad_sequences(np.array(y_train), maxlen=maxlen)
    y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))

    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    return X_train, y_train, X_test

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    logging.info('loading data...')
    pickle_file = os.path.join('pickle', 'cged_hsk_cwe.pickle3')
    revs, W, word_idx_map, vocab, maxlen = pickle.load(open(pickle_file, 'rb'))
    logging.info('data loaded!')

    X_train, y_train, X_test = make_idx_data(revs, word_idx_map, maxlen=maxlen)


    num_words = W.shape[0]
    logging.info("number of word vector [num_words]: %d" % num_words)

    embdding_dim = W.shape[1]               # 400
    logging.info("dimension num of word vector [embdding_dim]: %d" % embdding_dim)

    # --------------
    # 1. Regular CRF
    # --------------

    print('==== training CRF ====')

    model = Sequential()
    model.add(Embedding(num_words, embdding_dim, mask_zero=True, weights=[W], trainable=False))  # pre-trained embedding
    crf = CRF(len(error_dict), sparse_target=True)
    model.add(crf)

    model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])
    model.fit(X_train, y_train, epochs=EPOCHS, validation_data=[X_train, y_train])

    y_test_pred = model.predict(X_test).argmax(-1)
    print(y_test_pred)

    y_pred = []
    for i in range(len(y_test_pred)):
        line_data = y_test_pred[i]
        for j in range(len(line_data)):
            y_pred.append(line_data[j])

    print(y_pred.count(0))
    print(y_pred.count(1))
    print(y_pred.count(2))
    print(y_pred.count(3))
    print(y_pred.count(4))