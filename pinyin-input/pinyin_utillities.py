# Author: Zhen

import sys
import math
import string

def load_pinyin(pinyin_file, myNgram):
    used_charset = set() 
    for k in myNgram.ngram_counts.keys():
        prefixes = k.split(ngram.DELIM)
        if len(prefixes) == 1:
            used_charset.add(prefixes[0])
 
    i = 0
    j = 0
    char_set = []
    pinyin_set = []
    pinyin_index = {}
    c2p_mappings = {}
    for line in open(pinyin_file, 'r').readlines():
        fields = line.split()
        if len(fields) == 0: 
            continue
        char = fields[0]
        if not char in used_charset:
            continue
        pinyins = fields[1:]
        if len(pinyins) == 0:
            continue
        char_set.append(char)
        c2p_mappings[i] = set([])
        for t_pinyin in pinyins:
            pinyin = t_pinyin.strip(string.digits)
            if not pinyin_index.has_key(pinyin):
                pinyin_set.append(pinyin)
                pinyin_index[pinyin] = j
                p_index = j
                j += 1
            else:
                p_index = pinyin_index[pinyin]
            c2p_mappings[i].add(p_index)
            if not p2c_mappings.has_key(p_index):
                p2c_mappings[p_index] = [i]
            else:
                p2c_mappings[p_index].append(i)
        i += 1
    return char_set, pinyin_set, c2p_mappings, p2c_mappings

