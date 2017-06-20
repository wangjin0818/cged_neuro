from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import sys
import logging

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    input_file = os.path.join('vector', 'wiki.zh-tw.word.txt')
    output_file = os.path.join('vector', 'wiki.zh-tw.word.gensim.txt')

    with open(input_file, 'r') as fin:
        with open(output_file, 'w') as fout:
            for line in fin:
                fout.write(line.replace('\t', ' '))