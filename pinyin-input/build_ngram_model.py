import sys
import os
import re
import jieba
import ngram

hanzi_unicode_regexp = re.compile(u"[^\u4E00-\u9FA5]+")
eos_regex = re.compile(ur"[\uFF1F\uFF01\uFF1B\uFF1A?!;]")
EOS = u'\u3002'
BOS = u'\uFF1F'

def process_one_file(filename):
    corpus = []
    for line in open(filename).readlines():
        sentences = eos_regex.sub(u'\u3002', line.decode('utf-8'))
        for s in sentences.split(u'\u3002'):
            clean_s = hanzi_unicode_regexp.sub('', s)
            seg_list = jieba.cut(clean_s)
            for seg in seg_list:
                print seg.encode('utf-8')
                corpus.append(BOS)
                for char in seg:
                    utf_w = char.encode('utf-8')
                    corpus.append(char)
                corpus.append(EOS)

    return corpus
        
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


def build_initial_data(inputfile, output):
    corpus = process_files(inputfile)
    ngram.build_and_save_ngram(corpus, output, 2)

def build_smoothed_data(train_dict, heldout_dict, output):
    train_ngram = ngram.load_ngram(train_dict)
    heldout_ngram = ngram.load_ngram(heldout_dict)
    train_ngram.cross_validate(heldout_ngram, output)
    
if sys.argv[1] == "initial":
    build_initial_data(sys.argv[2], sys.argv[3])
elif sys.argv[1].strip() == "smooth":
    print "building smooth"
    build_smoothed_data(sys.argv[2], sys.argv[3], sys.argv[4])
