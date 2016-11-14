# Author: Zhen

import sys
import math
import string
import ngram
import utilities


def build_model(ngram_train_file, smooth_dict, pinyin_file, outputfile):
    myNgram = ngram.load_smoothed_ngram(ngram_train_file, smooth_dict)
    (char_set, pinyin_set, c2p_mappings, p2c_mappings) = 
        utilities.load_pinyin(pinyin_file, myNgram)
    print "size of char set ", len(char_set)
    output = open(outputfile, 'wc')
    output.writelines('hidden states:\n')
    output.writelines(' '.join(char_set) + '\n')
    output.writelines('items:\n')
    output.writelines(' '.join(pinyin_set) + '\n')
    n_chars = len(char_set)
    n_pinyin = len(pinyin_set)

    output.writelines('emission prob:\n')
    for i in range(n_chars):
        probs = []
        for j in range(n_pinyin):
            avg_prob = 1.0/len(mappings[i])
            if j in mappings[i]:
                probs.append(str(avg_prob))
            else:
                probs.append('0')
        output.writelines(' '.join(probs) + '\n')
    output.writelines('end emission prob\n')

    output.writelines('trans prob:\n')
    for i_char in char_set:
        probs = []
        total_prob = 0
        for j_char in char_set:
            prefix = [i_char.decode('utf-8'), j_char.decode('utf-8')]
            prob = math.exp(myNgram.get_prob(ngram.DELIM.join(prefix)))
            probs.append(prob)
            total_prob += prob
        # normalize
        normalized_probs = map(lambda x: x/total_prob, probs)
        output.writelines(' '.join(map(str, normalized_probs)) + '\n')
    output.writelines('end trans prob\n')

    output.writelines('initial prob:\n')
    probs = []
    total_prob = 0
    for char in char_set:
        prefix = [ngram.EOB, char.decode('utf-8')]
        prob = math.exp(myNgram.get_prob(ngram.DELIM.join(prefix)))
        total_prob += prob
        probs.append(prob)
    normalized_probs = map(lambda x: x/total_prob, probs)
#    normalized_probs = [1.0/len(char_set)] * len(char_set)
    output.writelines(' '.join(map(str, normalized_probs)) + '\n')
    output.close()


build_model(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
#char_set, pinyin_set, mappings = load_pinyin(sys.argv[1])
#for i in range(10):
#    char = char_set[i].decode('utf-8')
#    print char
#    for j in mappings[i]:
#        print pinyin_set[int(j)]
