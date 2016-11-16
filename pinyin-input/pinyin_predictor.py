# Author: Zhen Zhang
# -*- coding: utf-8 -*-

import sys
import hmm
import ngram
import ngram_utilities as nutil
import pinyin_utilities as pinyin_util

g_char_level_dict_path = "/Users/zhenzhang/Documents/ML/data/pinyin-input/models/sentence_split/training_data.txt"
PRUNE_LEN = 3

class InputMethod:

    def __init__(self, charset, pinyin_set, c2p, p2c, model):
        self.charset = charset
        self.pinyin_set = pinyin_set
        self.c2p = c2p
        self.p2c = p2c
        self.model = model
        self.load_char_level_ngram()

    def load_char_level_ngram(self):
        self.char_level_ngram = nutil.load_ngram(g_char_level_dict_path)

    def predict(self, pinyin):
        self.s = pinyin
        self.candidates = self.generate_candidates(pinyin)
        if len(self.candidates) == 0:
            print "No valid result"
            return
        max_score = -100
        res = None
        for candidate in self.candidates:
            chained, score = self.evaluate(candidate.decode("utf-8"))
            if float(score) > max_score:
                max_score = score
                res = candidate
        return res


    def generate_candidates(self, pinyin):
        candidates = []
        for p in pinyin.split():
            if not self.p2c.has_key(p):
                return []
            candidate_chars = self.p2c[p]
            candidates.append(candidate_chars)
        # enumerate to generate all candidates
        
        if len(candidates) == 0:
            return []
        cur_results = []
        for c in candidates[0]:
            cur_results.append([c])
        results = self.enumerate_candidate(cur_results, candidates[1:])
        return map(lambda x: ''.join(x), results)

    def enumerate_candidate(self, cur_results, remaining):
        if len(remaining) == 0:
            return cur_results
        results = []
        next_chars = remaining[0]
        for res in cur_results:
            if len(res) >= PRUNE_LEN and self.prune(res):
                continue 
            for c in next_chars: 
                    results.append(res + [c])
        return self.enumerate_candidate(results, remaining[1:])

    def prune(self, prefix):
        if len(prefix) <= 1: 
            return False
        miss_count = 0
        for i in range(len(prefix)-1):
            if not self.char_level_ngram.ngram_counts.has_key(
                    ngram.DELIM.join(prefix[i:i+2])):
                miss_count += 1
                if miss_count >= 2:
                    #print "pruning because ", ngram.DELIM.join(prefix[i:i+2])
                    return True
        return False

    def evaluate(self, candidate):
        prepared = self.model.prepare_input_from_string(candidate)
        return self.model.predict(prepared)

def generate_predictor_instance(train_dict, smoothed_dict, pinyin_file):
        pred_ngram = nutil.load_ngram(train_dict)
        pred_ngram.load_smoothed_dict(smoothed_dict)
        charset, pinyin_set, c2p, p2c = \
            pinyin_util.load_pinyin(pinyin_file)
        return InputMethod(charset, pinyin_set, c2p, p2c, pred_ngram)
