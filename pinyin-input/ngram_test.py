# encoding=utf-8
import sys
import os
import re
import math
import jieba
import ngram_utilities as nutil
from optparse import OptionParser

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
                help="predict prob of an input file") 
parser.add_option("-s", "--smoothed", dest="smoothed_dict",
                help="load smoothed prob dict", metavar="DICT")


(options, args) = parser.parse_args()
#predict(sys.argv[1], sys.argv[2])
#build_dict_from_single_file(sys.argv[1])
#build_dict(sys.argv[1], sys.argv[2])
if options.is_build:
    if (options.filename and options.dict):
        nutil.build_dict_from_single_file(options.filename, options.dict)
    if (options.dir and options.dict):
        nutil.build_dict(options.dir, options.dict)
if options.crossvalidate:
    if options.train_dict and options.heldout_dict:
        train_ngram = nutil.load_ngram(options.train_dict)
        heldout_ngram = nutil.load_ngram(options.heldout_dict)
        train_ngram.cross_validate(heldout_ngram, options.dict)
if options.predict:
    if (options.filename and options.train_dict and options.smoothed_dict):
        train_ngram = nutil.load_ngram(options.train_dict)
        train_ngram.load_smoothed_dict(options.smoothed_dict)
        paragraph = nutil.process_one_file(options.filename)
        print train_ngram.predict(paragraph)
