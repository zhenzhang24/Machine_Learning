# encoding=utf-8
import sys
import os
import re
import math
import jieba
import ngram
from optparse import OptionParser

def load_ngram(filename):
    ngram = Ngram()
    for line in open(filename).readlines():
       fields = line.split(' (') 
       prefix = fields[0]
       probs = fields[1].strip(')\n').split(', ')
       count = int(probs[0])
       prob = float(probs[1])
       l = len(prefix.split(DELIM))
       ngram.set_prob_dict(prefix, count, prob)
    return ngram

def load_smoothed_ngram(filename, smoothed_file):
    ngram = load_ngram(filename)
    ngram.load_smoothed_dict(smoothed_file)
    return ngram

def process_one_file(filename):
    f = open(filename)
    corpus = [ngram.EOB]
    for line in f.readlines():
        def etl(s):
            s2 = ngram.eos_regex.sub(u'\u3002', s, 10)
            s3 = ngram.sep_regex.sub(' ', s2, 10)
            return s3

        seg_list = filter(lambda x: len(x) > 0 and x.strip() != "", 
                map(etl, jieba.cut(line)))
        for w in seg_list:
                corpus.append(w)
                if w == ngram.EOS:
                    corpus.append(ngram.EOB)
    f.close()

    return corpus[:len(corpus)-1]


def process_files(root_dir):
    corpus = []
    i = 1
    for name in os.listdir(root_dir):
        if os.path.isfile(os.path.join(root_dir, name)):
            sys.stderr.write("process file %d %s\n"%(i,
                os.path.join(root_dir, name)))
            corpus.extend(process_one_file(os.path.join(root_dir, name)))
            i += 1
    return corpus

def build_and_save_ngram(corpus, output, n=3):
    ngram = Ngram()
    ngram.build_ngram(corpus, n)
    outputfile = open(output, 'wc')
    for k,v in ngram.ngram_probs.items():
        outputfile.write('%s %s\n'%(k.encode('utf-8'), str(v)))

def build_dict(old_dir, output, n=3):
    corpus = process_files(old_dir)
    build_and_save_ngram(corpus, output, n)

def build_dict_from_single_file(filename, output, n=3):
    corpus = process_one_file(filename)
    build_and_save_ngram(corpus, output, n)

def predict(dictfile, inputfile):
    ngram = load_ngram(dictfile)
    paragraph = process_one_file(inputfile)
    (chain, prob) = ngram.predict(paragraph)
    print prob
