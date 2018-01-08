from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import sys
import logging

import pandas as pd

error_idx = {
    0: 'C',
    1: 'R',
    2: 'M',
    3: 'S',
    4: 'W'
}

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    result_file = os.path.join('result', 'cged16_hsk_result.txt')
    result = pd.read_table(result_file, sep='\t', header=None, quoting=3)

    texts = result[1]
    sids = result[0]

    # with open(os.path.join('result', 'test_result.txt'), 'w') as my_file:
    my_file = open(os.path.join('result', 'test_result.txt'), 'w')

    for i in range(len(texts)):
        sent = texts[i]
        sid = sids[i]

        label = [int(w) for w in sent.split(',')]
        # for j in range(len(sent)):
        #     if sent[j] != 0:
        #         label.append(y_test_pred[i][j])

        error_flag = False
        is_correct = False

        current_error = 0
        start_pos = 0
        end_pos = 0

        for k in range(len(label)):
            # if sid == '200304131525200072_2_5x1':
            #         print(label[k])

            if label[k] != 0 and error_flag == False:
                error_flag = True
                start_pos = k + 1
                current_error = label[k]
                is_correct = True

                if sid == '200304131525200072_2_5x1':
                    print(label[k])
                    # print(label[k+1])
                    print(current_error)


            # if sid == '200304131525200072_2_5x1' and label[k] != current_error:
            #     print(label[k])

            if error_flag == True and label[k] != current_error and label[k] == 0:
                end_pos = k
                my_file.write('%s, %d, %d, %s\n' % (sid, start_pos, end_pos, error_idx[current_error]))

                error_flag = False
                current_error = 0

                if sid == '200304131525200072_2_5x1':
                    print(current_error)
                    print('%s, %d, %d, %s\n' % (sid, start_pos, end_pos, error_idx[current_error]))

            if error_flag == True and label[k] != current_error and label[k] != 0:
                end_pos = k
                my_file.write('%s, %d, %d, %s\n' % (sid, start_pos, end_pos, error_idx[current_error]))

                start_pos = k + 1
                current_error = label[k]

        if is_correct == False:
            my_file.write('%s, correct\n' % (sid))

        # if i % 100 == 0:
        #     logging.info('processed: %d/%d' % (i, len(sids)))

