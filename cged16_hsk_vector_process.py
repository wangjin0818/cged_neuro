from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import logging
import codecs
import random
import pickle
import gensim

import numpy as np
import pandas as pd

from xml.dom import minidom
from collections import defaultdict

error_dict = {
    'R': 1,
    'M': 2,
    'S': 3,
    'W': 4
}

def hsk_position_train_serialize(file_name):
    logging.info('Loading training data from %s' % (file_name))

    with codecs.open(file_name, 'r') as my_file:
        DOMTree = minidom.parse(my_file)

    docs = DOMTree.documentElement.getElementsByTagName('DOC')
    
    ret_text, ret_label = [], []
    len_text = []

    for doc in docs:
        text = doc.getElementsByTagName('TEXT')[0].childNodes[0].nodeValue.replace('\n', '')
        text_id = doc.getElementsByTagName('TEXT')[0].getAttribute('id')

        errs = doc.getElementsByTagName('ERROR')
        locate_dict = {}
        for err in errs:
            start_off = err.getAttribute('start_off')
            end_off = err.getAttribute('end_off')
            err_type = err.getAttribute('type')

            for i in range(int(start_off) - 1, int(end_off)):
                locate_dict[i] = error_dict[err_type]

        text_array = []
        label_array = []

        len_text.append(len(text))
        for i in range(len(text)):
            if i in locate_dict:
                text_array.append(text[i])
                label_array.append(locate_dict[i])
            else:
                text_array.append(text[i])
                label_array.append(0)

        ret_text.append(text_array)
        ret_label.append(label_array)

    return ret_text, ret_label

def hsk_position_test_serialize(file_name):
    logging.info('Loading testing data from %s' % (file_name))

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

    return revs, vocab_dict

def load_bin_vec(model, vocab):
    word_vecs = {}
    unk_words = 0

    for word in vocab.keys():
        try:
            word_vec = model[word]
            word_vecs[word] = word_vec
        except:
            unk_words = unk_words + 1
            # print(word)
    
    logging.info('unk words: %d' % (unk_words))
    return word_vecs

def get_W(word_vecs, k=200):
    vocab_size = len(word_vecs)
    word_idx_map = dict()

    W = np.zeros(shape=(vocab_size+2, k), dtype=np.float32)
    W[0] = np.zeros((embedding_dim, ))
    W[1] = np.random.uniform(-0.25, 0.25, k)

    i = 2
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i = i + 1
    return W, word_idx_map

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    train_file = os.path.join('corpus', 'nlptea16cged_release1.0', 'Training', 'CGED16_HSK_TrainingSet.txt')
    train_text, train_label = hsk_position_train_serialize(train_file)
    
    test_file = os.path.join('corpus', 'nlptea16cged_release1.0', 'Test', 'CGED16_HSK_Test_Input.txt')
    test_sid, test_text = hsk_position_test_serialize(test_file)

    revs, vocab = build_data_train_test(train_text, train_label, test_sid, test_text)
    max_l = np.max(pd.DataFrame(revs)['num_words'])
    mean_l = np.mean(pd.DataFrame(revs)['num_words'])
    logging.info('data loaded!')
    logging.info('number of sentences: ' + str(len(revs)))
    logging.info('vocab size: ' + str(len(vocab)))
    logging.info('max sentence length: ' + str(max_l))
    logging.info('mean sentence length: ' + str(mean_l))

    model_file = os.path.join('vector', 'wiki.zh-cn.word.gensim.txt')
    model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=False)
    logging.info('model vector size: %d' % (model.vector_size))

    w2v = load_bin_vec(model, vocab)
    logging.info('word2vec loaded!')
    logging.info('num words in word2vec: ' + str(len(w2v)))

    embedding_dim = model.vector_size
    W, word_idx_map = get_W(w2v, k=model.vector_size)
    logging.info('extracted index from word2vec! ')

    pickle_file = os.path.join('pickle', 'cged_hsk_cwe.pickle3')
    pickle.dump([revs, W, word_idx_map, vocab, max_l], open(pickle_file, 'wb'))
    logging.info('dataset created!')

