from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import sys
import logging
import pickle
import codecs

from xml.dom import minidom
from collections import defaultdict

error_dict = {
    'R': 1,
    'M': 2,
    'S': 3,
    'W': 4
}

def hsk_position_train_serialize(file_name, output_file_name):
    logging.info('Loading test data from %s' % (file_name))

    with codecs.open(file_name, 'r') as my_file:
        DOMTree = minidom.parse(my_file)

    docs = DOMTree.documentElement.getElementsByTagName('DOC')
    
    ret_text, ret_label = [], []
    len_text = []

    with codecs.open(output_file_name, 'w', 'utf8') as my_file:
        for doc in docs:
            text = doc.getElementsByTagName('TEXT')[0].childNodes[0].nodeValue.replace('\n', '').replace(' ', '')
            text_id = doc.getElementsByTagName('TEXT')[0].getAttribute('id')

            errs = doc.getElementsByTagName('ERROR')
            locate_dict = {}
            for err in errs:
                start_off = err.getAttribute('start_off')
                end_off = err.getAttribute('end_off')
                err_type = err.getAttribute('type')

                for i in range(int(start_off) - 1, int(end_off)):
                    # locate_dict[i] = error_dict[err_type]
                    locate_dict[i] = err_type

            text_array = []
            label_array = []

            len_text.append(len(text))
            for i in range(len(text)):
                if i in locate_dict:
                    text_array.append(text[i])
                    label_array.append(error_dict[locate_dict[i]])
                    my_file.write('%s\t%d\t%s\t%s\n' % (text_id, i, text[i], locate_dict[i]))
                else:
                    text_array.append(text[i])
                    label_array.append(0)
                    my_file.write('%s\t%d\t%s\t%s\n' % (text_id, i, text[i], u'C'))


            ret_text.append(text_array)
            ret_label.append(label_array)

    # statistic labels
    statistic_label = []
    for i in range(len(ret_label)):
        line_data = ret_label[i]
        for j in range(len(line_data)):
            statistic_label.append(line_data[j])

    print('Training statistic: ')
    print('Total len: %d' % (len(statistic_label)))
    print('C position: %d, ratio: %f' % (statistic_label.count(0), float(statistic_label.count(0)) / float(len(statistic_label))))
    print('R position: %d, ratio: %f' % (statistic_label.count(1), float(statistic_label.count(1)) / float(len(statistic_label))))
    print('M position: %d, ratio: %f' % (statistic_label.count(2), float(statistic_label.count(2)) / float(len(statistic_label))))
    print('S position: %d, ratio: %f' % (statistic_label.count(3), float(statistic_label.count(3)) / float(len(statistic_label))))
    print('W position: %d, ratio: %f' % (statistic_label.count(4), float(statistic_label.count(4)) / float(len(statistic_label))))

    return ret_text, ret_label

def hsk_position_test_serialize(file_name, output_test_file):
    logging.info('Loading test data from %s' % (file_name))

    ret_sid = []; ret_text = []
    with codecs.open(file_name, 'r', 'utf-8') as my_file:
        with codecs.open(output_test_file, 'w', 'utf-8') as out_file:
            for line in my_file.readlines():
                line = line.strip().split('\t')
                sid = line[0].replace('(sid=', '').replace(')', '')
                candi_text = line[1].replace(' ', '')
                text = [w for w in candi_text]
                for i in range(len(candi_text)):
                    out_file.write('%s\t%d\t%s\n' % (sid, i, candi_text[i]))

                ret_sid.append(sid)
                ret_text.append(text)

    return ret_sid, ret_text

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    train_file = os.path.join('corpus', 'nlptea16cged_release1.0', 'Training', 'CGED16_HSK_TrainingSet.txt')
    output_train_file = os.path.join('crf_file', 'cged16_hsk_word', 'train.txt')
    train_text, train_label = hsk_position_train_serialize(train_file, output_train_file)

    test_file = os.path.join('corpus', 'nlptea16cged_release1.0', 'Test', 'CGED16_HSK_Test_Input.txt')
    output_test_file = os.path.join('crf_file', 'cged16_hsk_word', 'test.txt')
    test_sid, test_text = hsk_position_test_serialize(test_file, output_test_file)

