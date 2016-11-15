# Author: Zhen

import sys
import math
import ngram
import string

def load_pinyin(pinyin_file):
    i = 0
    j = 0
    char_set = []
    pinyin_set = []
    c2p_mappings = {}
    p2c_mappings = {}
    for line in open(pinyin_file, 'r').readlines():
        fields = line.split()
        if len(fields) == 0: 
            continue
        char = fields[0]
        pinyins = fields[1:]
        if len(pinyins) == 0:
            continue
        char_set.append(char)
        c2p_mappings[i] = set([])
        for t_pinyin in pinyins:
            pinyin = t_pinyin.strip(string.digits)
            if not c2p_mappings.has_key(pinyin):
                pinyin_set.append(pinyin)
                c2p_mappings[char] = set([pinyin])
            else:
                c2p_mappings[char].add(pinyin)
            if not p2c_mappings.has_key(pinyin):
                p2c_mappings[pinyin] = set([char])
            else:
                p2c_mappings[pinyin].add(char)
        i += 1
    return char_set, pinyin_set, c2p_mappings, p2c_mappings
