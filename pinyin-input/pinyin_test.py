# Author: Zhen Zhang

import sys
import pinyin_predictor as ppredictor

predictor = ppredictor.generate_predictor_instance(sys.argv[1], 
        sys.argv[2], sys.argv[3])

print "Predicting"
while 1:
    line = sys.stdin.readline().rstrip('\n')
    result = predictor.predict(line)
    print result
