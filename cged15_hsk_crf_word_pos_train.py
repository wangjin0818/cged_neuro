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

from utils import pos_to_sequence_crf

error_dict = {
    'Missing': 'M',     ## M
    'Selection': 'S',   ## S
    'Redundant': 'R',   ## R
    'Disorder': 'W'     ## W
}

import opencc
cc = opencc.OpenCC('t2s')

def hsk_position_train_serialize(file_name):
    with codecs.open(file_name, 'r') as my_file:
        DOMTree = minidom.parse(my_file)

    docs = DOMTree.documentElement.getElementsByTagName('DOC')
    
    ret_text, ret_label = [], []
    len_text = []

    my_file = codecs.open(os.path.join('crf_file', 'cged16_hsk_word_pos', 'cged15_train_file.txt'), 'w', 'utf8')
    for doc in docs:
        text = doc.getElementsByTagName('SENTENCE')[0].childNodes[0].nodeValue.replace('\n', '')
        text_id = doc.getElementsByTagName('SENTENCE')[0].getAttribute('id')

        err = doc.getElementsByTagName('TYPE')[0].childNodes[0].nodeValue
        start_off =  doc.getElementsByTagName('MISTAKE')[0].getAttribute('start_off')
        end_off =  doc.getElementsByTagName('MISTAKE')[0].getAttribute('start_off')

        locate_dict = {}
        for i in range(int(start_off) - 1, int(end_off)):
            locate_dict[i] = err

        text_array = []
        label_array = []

        text = cc.convert(text)
        word_seq, pos_seq = pos_to_sequence_crf(text)
        print(text)
        # print(word_seq, pos_seq)

        for i in range(len(word_seq)):
            if i in locate_dict.keys():
                text_array.append(word_seq[i])
                label_array.append(error_dict[locate_dict[i]])
                my_file.write('%s\t%d\t%s\t%s\t%s\n' % (text_id, i, word_seq[i], pos_seq[i], error_dict[locate_dict[i]]))
                # print('%s\t%d\t%s\t%s\t%s\n' % (text_id, i, word_seq[i], pos_seq[i], error_dict[locate_dict[i]]))

                # error = locate_dict[i] + '-' + pos_seq[i]
                # label_array.append(error)
                # error_pos_dict[error] = error_pos_dict[error] + 1
            else:
                text_array.append(word_seq[i])
                label_array.append(0)
                my_file.write('%s\t%d\t%s\t%s\t%s\n' % (text_id, i, word_seq[i], pos_seq[i], u'C'))
                # print('%s\t%d\t%s\t%s\t%s\n' % (text_id, i, word_seq[i], pos_seq[i], u'C'))

                # error = 'C-' + pos_seq[i]
                # label_array.append(error)

        # my_file.write('%s\t%s\n' % (','.join(text_array), ','.join(label_array)))

    return ret_text, ret_label

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    train_file = os.path.join('corpus', 'nlptea15cged_release1.0', 'Training', 'NLPTEA15_CGED_Training.sgml')
    train_text, train_label = hsk_position_train_serialize(train_file)