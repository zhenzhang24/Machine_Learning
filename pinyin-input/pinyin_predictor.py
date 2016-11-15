# Author: Zhen Zhang

import sys
import hmm
import ngram
import ngram_utilities as nutil
import pinyin_utilities as pinyin_util

class InputMethod:

    def __init__(self, charset, pinyin_set, c2p, p2c, model):
        self.charset = charset
        self.pinyin_set = pinyin_set
        self.c2p = c2p
        self.p2c = p2c
        self.model = model


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
        
        results = self.enumerate_candidate(candidates)
        return map(lambda x: ''.join(x), results)

    def enumerate_candidate(self, candidates):
        if len(candidates) == 0:
            return []
        first_chars = candidates[0]
        rest = self.enumerate_candidate(candidates[1:])
        results = []
        for c in first_chars:
            if len(rest) == 0:
                results.append([c])
                continue
            for r in rest:
                results.append([c] + r)
        return results


    def evaluate(self, candidate):
        prepared = self.model.prepare_input_from_string(candidate)
        return self.model.predict(prepared)

def generate_predictor_instance(train_dict, smoothed_dict, pinyin_file):
        pred_ngram = nutil.load_ngram(train_dict)
        pred_ngram.load_smoothed_dict(smoothed_dict)
        charset, pinyin_set, c2p, p2c = \
            pinyin_util.load_pinyin(pinyin_file)
        return InputMethod(charset, pinyin_set, c2p, p2c, pred_ngram)
