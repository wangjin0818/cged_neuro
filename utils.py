#coding:utf8
import os
import sys
import logging

from nltk.tokenize.stanford_segmenter import StanfordSegmenter
from nltk.tag import StanfordPOSTagger

java_path = "C:\\Program Files\\Java\\jdk1.8.0_73\\bin\\java.exe"
os.environ['JAVAHOME'] = java_path

segmenter = StanfordSegmenter(path_to_jar="E:\\lib\\stanford-segmenter-2017-06-09\\stanford-segmenter-3.8.0.jar", 
    path_to_slf4j="E:\\lib\\stanford-segmenter-2017-06-09\\slf4j-api.jar",
    path_to_sihan_corpora_dict="E:\\lib\\stanford-segmenter-2017-06-09\\data", 
    path_to_model="E:\\lib\\stanford-segmenter-2017-06-09\\data\\pku.gz", 
    path_to_dict="E:\\lib\\stanford-segmenter-2017-06-09\\data\\dict-chris6.ser.gz"
)

postagger = StanfordPOSTagger(path_to_jar="E:\\lib\\stanford-postagger-full-2017-06-09\\stanford-postagger.jar",
    model_filename='E:\\lib\\stanford-postagger-full-2017-06-09\\models\\chinese-distsim.tagger',
)

def pos_to_sequence(sent, segmenter=segmenter, postagger=postagger):
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
            tag_seq.append(tag)

    return word_seq, tag_seq

def pos_to_sequence_crf(sent, segmenter=segmenter, postagger=postagger):
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
