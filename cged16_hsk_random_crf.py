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

error_idx = {
    0: 'C',
    1: 'R',
    2: 'M',
    3: 'S',
    4: 'W'
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
    X_train, X_test, y_train, sid_test = [], [], [], []
    for rev in revs:
        sent = get_idx_from_sent(rev['text'], word_idx_map)

        if rev['option'] == 'train':
            y = rev['label']

            X_train.append(sent)
            y_train.append(y)

        elif rev['option'] == 'test':
            X_test.append(sent)
            sid_test.append(rev['sid'])

    X_train = sequence.pad_sequences(np.array(X_train), maxlen=maxlen)
    X_test = sequence.pad_sequences(np.array(X_test), maxlen=maxlen)
    # X_valid = sequence.pad_sequences(np.array(X_valid), maxlen=maxlen)
    y_train = sequence.pad_sequences(np.array(y_train), maxlen=maxlen)
    y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))

    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    return X_train, y_train, X_test, sid_test

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    logging.info('loading data...')
    pickle_file = os.path.join('pickle', 'cged_hsk_random.pickle3')
    revs, vocab, word_idx_map, maxlen = pickle.load(open(pickle_file, 'rb'))
    logging.info('data loaded!')

    X_train, y_train, X_test, sid_test = make_idx_data(revs, word_idx_map, maxlen=maxlen)

    # --------------
    # 1. Regular CRF
    # --------------

    print('==== training CRF ====')

    model = Sequential()
    model.add(Embedding(len(vocab), EMBEDDING_DIM, mask_zero=True))  # Random embedding
    crf = CRF(len(error_dict), sparse_target=True)
    model.add(crf)

    model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])
    model.fit(X_train, y_train, epochs=EPOCHS, validation_data=[X_train, y_train])

    y_test_pred = model.predict(X_test).argmax(-1)

    # result_file = os.path.join('result', 'cged16_hsk_result.txt')
    # with open(result_file, 'w') as my_file:
    #     for i in range(len(X_test)):
    #         sent = X_test[i]
    #         sid = sid_test[i]

    #         label = []
    #         for j in range(len(sent)):
    #             if sent[j] != 0:
    #                 label.append(str(y_test_pred[i][j]))

    #         my_file.write('%s\t%s\n' % (sid, ','.join(label)))

    ## statistic result
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

    print(len(X_test))
    print(len(sid_test))

    final_result_file = os.path.join('result', 'cged16_hsk_random_crf.txt')
    with open(final_result_file, 'w') as my_file:
        for i in range(len(X_test)):
            sent = X_test[i]
            sid = sid_test[i]

            label = []
            for j in range(len(sent)):
                if sent[j] != 0:
                    label.append(y_test_pred[i][j])

            error_flag = False
            is_correct = False

            current_error = 0
            start_pos = 0
            end_pos = 0
            error_dict = {}
            for k in range(len(label)):
                if label[k] != 0 and error_flag == False:
                    error_flag = True
                    start_pos = k + 1
                    current_error = label[k]
                    is_correct = True

                if error_flag == True and label[k] != current_error and label[k] == 0:
                    end_pos = k
                    my_file.write('%s, %d, %d, %s\n' % (sid, start_pos, end_pos, error_idx[current_error]))

                    error_flag = False
                    current_error = 0

                if error_flag == True and label[k] != current_error and label[k] != 0:
                    end_pos = k
                    my_file.write('%s, %d, %d, %s\n' % (sid, start_pos, end_pos, error_idx[current_error]))

                    start_pos = k + 1
                    current_error = label[k]

            if is_correct == False:
                my_file.write('%s, correct\n' % (sid))

            if i % 100 == 0:
                logging.info('processed: %d/%d' % (i, len(sid_test)))

    