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

from nltk.tokenize.stanford_segmenter import StanfordSegmenter
from nltk.tag import StanfordPOSTagger

java_path = "C:\\Program Files\\Java\\jdk1.8.0_73\\bin\\java.exe"
os.environ['JAVAHOME'] = java_path

error_dict = {
    'R': 1,
    'M': 2,
    'S': 3,
    'W': 4
}

segmenter = StanfordSegmenter(path_to_jar="E:\\lib\\stanford-segmenter-2017-06-09\\stanford-segmenter-3.8.0.jar", 
    path_to_slf4j="E:\\lib\\stanford-segmenter-2017-06-09\\slf4j-api.jar",
    path_to_sihan_corpora_dict="E:\\lib\\stanford-segmenter-2017-06-09\\data", 
    path_to_model="E:\\lib\\stanford-segmenter-2017-06-09\\data\\pku.gz", 
    path_to_dict="E:\\lib\\stanford-segmenter-2017-06-09\\data\\dict-chris6.ser.gz"
)

postagger = StanfordPOSTagger(path_to_jar="E:\\lib\\stanford-postagger-full-2017-06-09\\stanford-postagger.jar",
    model_filename='E:\\lib\\stanford-postagger-full-2017-06-09\\models\\chinese-distsim.tagger',
)

def pos_to_sequence(sent, segmenter, postagger):
    seg_sent = segmenter.segment(sent)

    tag_sent = postagger.tag(seg_sent.strip().split())

    word_seq = []
    tag_seq = []
    for item in tag_sent:
        word_and_pos = item[1].split('#')
        word = word_and_pos[0]
        tag = word_and_pos[1]

        for i in range(len(word)):
            word_seq.append(word[i])
            if i == 0:
                tag_seq.append('B-' + tag)
            else:
                tag_seq.append('I-' + tag)

    return word_seq, tag_seq

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

            text_seq, pos_seq = pos_to_sequence(text, segmenter, postagger)

            len_text.append(len(text))
            for i in range(len(text_seq)):
                if i in locate_dict:
                    text_array.append(text[i])
                    label_array.append(error_dict[locate_dict[i]])
                    my_file.write('%s\t%d\t%s\t%s\t%s\n' % (text_id, i, text_seq[i], pos_seq[i], locate_dict[i]))
                else:
                    text_array.append(text[i])
                    label_array.append(0)
                    my_file.write('%s\t%d\t%s\t%s\t%s\n' % (text_id, i, text_seq[i], pos_seq[i], u'C'))

            print(text)
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

                text_seq, pos_seq = pos_to_sequence(candi_text, segmenter, postagger)

                text = [w for w in candi_text]
                for i in range(len(text_seq)):
                    out_file.write('%s\t%d\t%s\t%s\n' % (sid, i, text_seq[i], pos_seq[i]))

                print(candi_text)
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
    output_train_file = os.path.join('crf_file', 'cged16_hsk_word_pos', 'train.txt')
    train_text, train_label = hsk_position_train_serialize(train_file, output_train_file)

    test_file = os.path.join('corpus', 'nlptea16cged_release1.0', 'Test', 'CGED16_HSK_Test_Input.txt')
    output_test_file = os.path.join('crf_file', 'cged16_hsk_word_pos', 'test.txt')
    test_sid, test_text = hsk_position_test_serialize(test_file, output_test_file)

