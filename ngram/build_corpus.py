# encoding=utf-8
import sys
import os
import re
import math
import jieba
from optparse import OptionParser

eos_regex = re.compile(ur"[\uFF1F\uFF01\uFF1B\uFF1A?!;]")
sep_regex = re.compile(ur"[\u3001\uFF0C\u300C\u300D\u300E\u300F\u2018\u2019\u201C\u201D\uFF08\uFF09\u3014\u3015\u3010\u3011\u2013\u2014\u2026\uFF0E\u300A\u300B\u3008\u3009\|():\~\#\$\%\^\&\*\.\-\[\]\\]")
EOS = u'\u3002'
EOB = u'\uFF1F'
INFINITY_SMALL = -10000000
DELIM = '|'

class Ngram:

    def __init__(self, n=3):
        self.corpus = []
        self.n = n
        self.ngram_counts = {}
        self.ngram_probs = {}
        self.heldout_probs = {}
        self.ngram = []
        self.total_ngram_counts = [0] * self.n

    def build_ngram(self, corpus):
        self.corpus = corpus
        self.build_ngram_counts()
        self.build_ngram_prob()

    def build_ngram_counts(self):
        word_list = []
        for w in self.corpus:
            for i in range(self.n):
                if i < len(word_list):
                    previous_gram = word_list[i] 
                else:
                    word_list.append([])
                    previous_gram = None
                if previous_gram:
                    if len(previous_gram) == i+1:
                        cur_gram = previous_gram[1:]
                    else:
                        cur_gram = previous_gram
                    cur_gram.append(w)
                else:
                    cur_gram = previous_gram = [w]
                word_list[i] = previous_gram = cur_gram
                if len(cur_gram) == (i + 1):
                    self.update_count(i,cur_gram)
            if w == EOS:
                word_list = []

    def build_ngram_prob(self):
        for ngram_k, count in self.ngram_counts.items():
            ngram = ngram_k.split(DELIM)
            n = len(ngram)
            if n == 0: continue
            prefix = DELIM.join(ngram[0:n-1])
            if n == 1:
                self.ngram_probs[ngram_k] = (self.ngram_counts[ngram_k], 
                        float(self.ngram_counts[ngram_k]) / \
                        self.total_ngram_counts[0])
            else:
                self.ngram_probs[ngram_k] = (self.ngram_counts[ngram_k],
                        float(self.ngram_counts[ngram_k]) /\
                        self.ngram_counts[prefix])

    def update_count(self, n, w):
        key = DELIM.join(w)
        if self.ngram_counts.has_key(key):
            self.ngram_counts[key] += 1
        else:
            self.ngram_counts[key] = 1
        self.total_ngram_counts[n] += 1

    def set_prob_dict(self, prefix, count, prob):
        self.ngram_counts[prefix] = count
        self.ngram_probs[prefix] = (count, prob)

    def get_all_prefixes(self, ngram):
        prefixes = ngram.split(DELIM)
        prefix_ngrams = []
        for i in range(len(prefixes)):
            prefix_ngrams.append(DELIM.join(prefixes[i:]))
        return prefix_ngrams

    def get_prob(self, ngram):
        prefix_ngrams = self.get_all_prefixes(ngram)
        res = 0
        for p in prefix_ngrams:
            prefix = p.encode('utf-8')
            if self.ngram_probs.has_key(prefix):
                print "find prefix ", prefix
                res += 0.333 * math.log(self.ngram_probs[prefix][1])
            else:
                print "not found prefix ", prefix
                res += 0.333 * self.heldout_probs[0][3]
        return res

    def MLE(self, cutted):
        chained = []
        cur_ngram = []
        for w in cutted:
            if w == EOB:
                cur_ngram = [w]
                continue
            if len(cur_ngram) == self.n:
                cur_ngram = cur_ngram[1:]
            cur_ngram.append(w)

            chained.append(DELIM.join(cur_ngram))

        total_prob = 0
        for k in chained:
            prob = self.get_prob(k)
            if prob == 0:
                print "not found:", k.encode('utf-8')
                return chained, INFINITY_SMALL
            total_prob += prob
        return chained, total_prob

    def group_by_count(self):
        self.ngram_by_freq = {}
        for k, freq in self.ngram_counts.items():
            if not self.ngram_by_freq.has_key(freq):
                self.ngram_by_freq[freq] = []
            self.ngram_by_freq[freq].append(k)


    def interpolate(self):
        self.ngram_by_len = {}
        for i in range(self.n):
            self.ngram_by_len[i] = []
        for k, v in self.ngram_probs.items():
            n = k.split(DELIM)
            self.ngram_by_len[n].append(k)
        for w in self.ngram_by_len[0]:
            self.ngram_probs[w] = self.ngram_probs[w]

    # turing good: Tr/T * 1/Nr
    def cross_validate(self, heldout_ngram, output):
        total = 0
        heldout_counts = {0:0}
        ref_set = heldout_ngram.ngram_counts
        self.ngram_types_by_freq = {}
        for ngram, freq in self.ngram_counts.items():
            if not heldout_counts.has_key(freq):
                heldout_counts[freq] = 0
            if ref_set.has_key(ngram):
                heldout_counts[freq] += ref_set[ngram]
            if not self.ngram_types_by_freq.has_key(freq):
                self.ngram_types_by_freq[freq] = 0
            self.ngram_types_by_freq[freq] += 1

        self.ngram_types_by_freq[0] = 0
        for ngram, freq in ref_set.items():
            total += freq
            if not self.ngram_counts.has_key(ngram):
                heldout_counts[0] += freq
                self.ngram_types_by_freq[0] += 1

        outputfile = open(output, 'wc')
        for freq, occurrence in heldout_counts.items():
            Nr = self.ngram_types_by_freq[freq]
            self.heldout_probs[freq] = (occurrence, total, Nr,  
                math.log(occurrence) - math.log(total * Nr))
            outputfile.write('freq:%d Tr:%d T:%d Nr:%d P:%f\n'%
                    (freq, occurrence, total, Nr, self.heldout_probs[freq]))

    def load_smoothed_dict(self, dictname):
        f = open(dictname)
        for l in f.readlines():
            fields = l.split()
            freq = int(fields[0].strip('freq:'))
            Tr = int(fields[1].strip('Tr:'))
            T = int(fields[2].strip('T:'))
            Nr = int(fields[3].strip('Nr:'))
            P = float(fields[4].strip('P:'))
            self.heldout_probs[freq] = (Tr, T, Nr, P)
        f.close()

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

def process_one_file(filename):
    f = open(filename)
    corpus = [EOB]
    for line in f.readlines():
        def etl(s):
            s2 = eos_regex.sub(u'\u3002', s, 10)
            s3 = sep_regex.sub(' ', s2, 10)
            return s3

        seg_list = filter(lambda x: len(x) > 0 and x.strip() != "", 
                map(etl, jieba.cut(line)))
        for w in seg_list:
                corpus.append(w)
                if w == EOS:
                    corpus.append(EOB)
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

def build_dict(old_dir, output):
    corpus = process_files(old_dir)
    #corpus = process_one_file(sys.argv[1])
    ngram = Ngram()
    ngram.build_ngram(corpus)
    outputfile = open(output, 'wc')
    for k,v in ngram.ngram_probs.items():
        outputfile.write('%s %s\n'%(k.encode('utf-8'), str(v)))

def build_dict_from_single_file(filename, output):
    corpus = process_one_file(filename)
    ngram = Ngram()
    ngram.build_ngram(corpus)
    outputfile = open(output, 'wc')
    for k,v in ngram.ngram_probs.items():
        outputfile.write('%s %s\n'%(k.encode('utf-8'), str(v)))

def predict(dictfile, inputfile):
    ngram = load_ngram(dictfile)
    paragraph = process_one_file(inputfile)
    (chain, prob) = ngram.MLE(paragraph)
    print prob

parser = OptionParser()
parser.add_option("-b", "--build", 
                action='store_true', dest='is_build', 
                default=False,
                help="build training ngram") 
parser.add_option("-f", "--file", dest="filename",
                help="build ngram from single file", metavar="FILE")
parser.add_option("-d", "--dir", dest="dir",
                help="build ngram from dir", metavar="DIR")
parser.add_option("-i", "--dict", dest="dict",
                help="store dict", metavar="DICT")


parser.add_option("-c", "--crossvalidate", 
                action='store_true', dest='crossvalidate', 
                default=False,
                help="build heldout ngram") 
parser.add_option("-t", "--train", dest="train_dict",
                help="load training ngram", metavar="DICT")
parser.add_option("-o", "--heldout", dest="heldout_dict",
                help="load heldout ngram", metavar="DICT")

parser.add_option("-p", "--predict", 
                action='store_true', dest='predict', 
                default=False,
                help="predict MLE of an input file") 
parser.add_option("-s", "--smoothed", dest="smoothed_dict",
                help="load smoothed prob dict", metavar="DICT")


(options, args) = parser.parse_args()
#predict(sys.argv[1], sys.argv[2])
#build_dict_from_single_file(sys.argv[1])
#build_dict(sys.argv[1], sys.argv[2])
if options.is_build:
    if (options.filename and options.dict):
        build_dict_from_single_file(options.filename, options.dict)
    if (options.dir and options.dict):
        build_dict(options.dir, options.dict)
if options.crossvalidate:
    if options.train_dict and options.heldout_dict:
        train_ngram = load_ngram(options.train_dict)
        heldout_ngram = load_ngram(options.heldout_dict)
        train_ngram.cross_validate(heldout_ngram, options.dict)
if options.predict:
    if (options.filename and options.train_dict and options.smoothed_dict):
        train_ngram = load_ngram(options.train_dict)
        train_ngram.load_smoothed_dict(options.smoothed_dict)
        paragraph = process_one_file(options.filename)
        print train_ngram.MLE(paragraph)
