from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import sys
import logging
import codecs

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    # input_file = os.path.join('output', 'cged15_test_file.txt')
    # output_file = os.path.join('output', 'cged15_test_file_singlelabel.txt')

    # input_file = os.path.join('output', 'cged15_train_file.txt')
    # output_file = os.path.join('output', 'cged15_train_file_singlelabel.txt')

    input_file = os.path.join('output', 'cged16_hsk_train_file.txt')
    output_file = os.path.join('output', 'cged16_hsk_train_file_singlelabel.txt')

    with codecs.open(input_file, 'r', 'utf8') as in_file:
        with codecs.open(output_file, 'w', 'utf8') as out_file:
            for in_line in in_file.readlines():
                in_line = in_line.strip().split('\t', 1)
                out_labels = [label.split('-')[0] if (label.startswith('R') or label.startswith('S') \
                                                      or label.startswith('W')
                                                      or label.startswith('M'))
                                                  else label.split('-')[1] for label in in_line[1].split(',')]     

                out_file.write('%s\t%s\n' % (in_line[0], ','.join(out_labels)))
