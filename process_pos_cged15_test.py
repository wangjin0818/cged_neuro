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

from utils import pos_to_sequence

error_dict = {
    'Missing': 'M',
    'Selection': 'S',
    'Redundant': 'R',
    'Disorder': 'D'
}

import opencc
cc = opencc.OpenCC('t2s')

def hsk_position_train_serialize(file_name):
    with codecs.open(file_name, 'r') as my_file:
        DOMTree = minidom.parse(my_file)

    docs = DOMTree.documentElement.getElementsByTagName('DOC')
    
    ret_text, ret_label = [], []
    len_text = []

    my_file = codecs.open(os.path.join('output', 'cged15_test_file.txt'), 'w', 'utf8')
    for doc in docs:
        text = doc.getElementsByTagName('SENTENCE')[0].childNodes[0].nodeValue.replace('\n', '')
        text_id = doc.getElementsByTagName('SENTENCE')[0].getAttribute('id')

        err = doc.getElementsByTagName('TYPE')[0].childNodes[0].nodeValue
        start_off =  doc.getElementsByTagName('MISTAKE')[0].getAttribute('start_off')
        end_off =  doc.getElementsByTagName('MISTAKE')[0].getAttribute('start_off')

        locate_dict = {}
        for i in range(int(start_off) - 1, int(end_off)):
            locate_dict[i] = error_dict[err]

        text_array = []
        label_array = []

        print(text)
        text = cc.convert(text)

        word_seq, pos_seq = pos_to_sequence(text)
        print(text)

        for i in range(len(word_seq)):
            if i in locate_dict:
                text_array.append(word_seq[i])
                error = locate_dict[i] + '-' + pos_seq[i]
                label_array.append(error)
                # error_pos_dict[error] = error_pos_dict[error] + 1
            else:
                text_array.append(word_seq[i])
                error = 'C-' + pos_seq[i]
                label_array.append(error)

        my_file.write('%s\t%s\n' % (','.join(text_array), ','.join(label_array)))

    return ret_text, ret_label

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    test_file = os.path.join('corpus', 'nlptea15cged_release1.0', 'Test', 'NLPTEA15_CGED_Test.sgml')
    train_text, train_label = hsk_position_train_serialize(test_file)