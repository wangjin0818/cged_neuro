from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import sys
import logging
import pickle
import codecs

import pandas as pd

option = 4

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    result_file = os.path.join('crf_file', 'cged16_hsk_word_pos', 'result.txt')
    result = pd.read_table(result_file, sep='\t', header=None, quoting=3)

    final_result_file = os.path.join('crf_file', 'cged16_hsk_word_pos', 'final_result.txt')

    sids = set(result[0])
    error_num = result[result[option] != 'C'].count()
    logging.info('total error num: %d' % (error_num[0]))

    with codecs.open(final_result_file, 'w', 'utf8') as my_file:
        n = 1
        for sid in sids:
            result_data = result[result[0] == sid]
            # print(result_data[3][336896])
            error_flag = False
            is_correct = False

            current_error = 'C'
            start_pos = 0
            end_pos = 0
            error_dict = {}
            i = 0
            for index, row in result_data.iterrows():
                if row[option] != 'C' and error_flag == False:
                    error_flag = True
                    start_pos = i + 1
                    current_error = row[option]
                    is_correct = True

                if error_flag == True and row[option] != current_error and row[option] == 'C':
                    end_pos = i
                    my_file.write('%s, %d, %d, %s\n' % (sid, start_pos, end_pos, current_error))

                    error_flag = False
                    current_error = 'C'

                if error_flag == True and row[option] != current_error and row[option] != 'C':
                    end_pos = i
                    my_file.write('%s, %d, %d, %s\n' % (sid, start_pos, end_pos, current_error))

                    start_pos = i + 1
                    current_error = row[option]

                i = i + 1

            if is_correct == False:
                my_file.write('%s, correct\n' % (sid))

            if n % 100 == 0:
                logging.info('processed: %d/%d' % (n, len(sids)))
            n = n + 1



    


