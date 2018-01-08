from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import logging
import codecs
import random
import pickle

import numpy as np
import pandas as pd

from xml.dom import minidom
from collections import defaultdict

# error_dict = {
#     'R': 1,
#     'M': 2,
#     'S': 3,
#     'W': 4
# }

# def hsk_position_train_serialize(file_name):
#     with codecs.open(file_name, 'r') as my_file:
#         DOMTree = minidom.parse(my_file)

#     docs = DOMTree.documentElement.getElementsByTagName('DOC')
    
#     ret_text, ret_label = [], []
#     len_text = []

#     for doc in docs:
#         text = doc.getElementsByTagName('TEXT')[0].childNodes[0].nodeValue.replace('\n', '')
#         text_id = doc.getElementsByTagName('TEXT')[0].getAttribute('id')

#         errs = doc.getElementsByTagName('ERROR')
#         locate_dict = {}
#         for err in errs:
#             start_off = err.getAttribute('start_off')
#             end_off = err.getAttribute('end_off')
#             err_type = err.getAttribute('type')

#             for i in range(int(start_off) - 1, int(end_off)):
#                 locate_dict[i] = error_dict[err_type]

#         text_array = []
#         label_array = []

        

#         len_text.append(len(text))
#         for i in range(len(text)):
#             if i in locate_dict:
#                 text_array.append(text[i])
#                 label_array.append(locate_dict[i])
#             else:
#                 text_array.append(text[i])
#                 label_array.append(0)

#         ret_text.append(text_array)
#         ret_label.append(label_array)

#         # statistic_label = ret_label
#         # statistic_label = np.array(statistic_label)

#     statistic_label = []
#     for i in range(len(ret_label)):
#         line_data = ret_label[i]
#         for j in range(len(line_data)):
#             statistic_label.append(line_data[j])

#     print('Training statistic: ')
#     print('Total len: %d' % (len(statistic_label)))
#     print('C position: %d, ratio: %f' % (statistic_label.count(0), float(statistic_label.count(0)) / float(len(statistic_label))))
#     print('R position: %d, ratio: %f' % (statistic_label.count(1), float(statistic_label.count(1)) / float(len(statistic_label))))
#     print('M position: %d, ratio: %f' % (statistic_label.count(2), float(statistic_label.count(2)) / float(len(statistic_label))))
#     print('S position: %d, ratio: %f' % (statistic_label.count(3), float(statistic_label.count(3)) / float(len(statistic_label))))
#     print('W position: %d, ratio: %f' % (statistic_label.count(4), float(statistic_label.count(4)) / float(len(statistic_label))))

#     return ret_text, ret_label

def hsk_position_test_serialize(file_name):
    logging.info('Loading test data from %s' % (file_name))

    ret_sid = []; ret_text = []
    with codecs.open(file_name, 'r', 'utf-8') as my_file:
        for line in my_file.readlines():
            line = line.strip().split('\t')
            sid = line[0].replace('(sid=', '').replace(')', '')
            text = [w for w in line[1]]

            ret_sid.append(sid)
            ret_text.append(text)

    return ret_sid, ret_text

def build_data_train_test(train_text, train_label, test_sid, test_text):
    revs = []
    vocab_dict = defaultdict(int)

    for i in range(len(train_text)):
        text = train_text[i]
        label = train_label[i]
        words = set(text)

        for word in words:
            vocab_dict[word] = vocab_dict[word] + 1
        datum = {
            'text': text,
            'label': label,
            'num_words': len(text),
            'option': 'train'
        }
        revs.append(datum)

    for i in range(len(test_text)):
        text = test_text[i]
        sid = test_sid[i]
        words = set(text)

        for word in words:
            vocab_dict[word] = vocab_dict[word] + 1
        datum = {
            'sid': sid,
            'text': text,
            'num_words': len(text),
            'option': 'test'
        }
        revs.append(datum)

    vocab = ['<pad>', '<unk>'] + [w for w in vocab_dict.keys()]
    word_idx_map = {}
    for i in range(len(vocab)):
        word_idx_map[vocab[i]] = i

    return revs, vocab, word_idx_map

def load_train_data(train_file, train_text, train_label):
    with codecs.open(train_file, 'r', 'utf8') as my_file:
        for line in my_file.readlines():
            line = line.strip().split('\t')

            train_text.append(line[0].split(','))
            train_label.append(line[1].split(','))

    return train_text, train_label

def label_serilization(train_label):
    label_type = sum(train_label, [])
    label_type = list(set(label_type))

    error_to_idx = defaultdict(int)
    idx_to_error = defaultdict(str)
    for i in range(len(label_type)):
        error_to_idx[label_type[i]] = i + 1
        idx_to_error[i + 1] = label_type[i]

    new_train_label = []
    for i in range(len(train_label)):
        new_line = []
        for j in range(len(train_label[i])):
            new_line.append(error_to_idx[train_label[i][j]])

        new_train_label.append(new_line)

    return new_train_label, error_to_idx, idx_to_error

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    # train_file = os.path.join('corpus', 'nlptea16cged_release1.0', 'Training', 'CGED16_HSK_TrainingSet.txt')
    # train_text, train_label = hsk_position_train_serialize(train_file)

    train_text, train_label = [], []
    cged15_train_file = os.path.join('output', 'cged15_train_file_singlelabel.txt')
    train_text, train_label = load_train_data(cged15_train_file, train_text, train_label)

    cged15_test_file = os.path.join('output', 'cged15_test_file_singlelabel.txt')
    train_text, train_label = load_train_data(cged15_test_file, train_text, train_label)

    cged16_hsk_train_file = os.path.join('output', 'cged16_hsk_train_file_singlelabel.txt')
    train_text, train_label = load_train_data(cged16_hsk_train_file, train_text, train_label)

    train_label, error_to_idx, idx_to_error = label_serilization(train_label)
    
    test_file = os.path.join('corpus', 'nlptea16cged_release1.0', 'Test', 'CGED16_HSK_Test_Input.txt')
    test_sid, test_text = hsk_position_test_serialize(test_file)

    revs, vocab, word_idx_map = build_data_train_test(train_text, train_label, test_sid, test_text)
    max_l = np.max(pd.DataFrame(revs)['num_words'])
    mean_l = np.mean(pd.DataFrame(revs)['num_words'])
    logging.info('data loaded!')
    logging.info('number of sentences: ' + str(len(revs)))
    logging.info('vocab size: ' + str(len(vocab)))
    logging.info('max sentence length: ' + str(max_l))
    logging.info('mean sentence length: ' + str(mean_l))

    print(error_to_idx['R'], error_to_idx['S'], error_to_idx['M'], error_to_idx['W'])

    pickle_file = os.path.join('pickle', 'cged_hsk_singlelabel_random.pickle3')
    pickle.dump([revs, vocab, word_idx_map, max_l, error_to_idx, idx_to_error], open(pickle_file, 'wb'))
    logging.info('dataset created!')