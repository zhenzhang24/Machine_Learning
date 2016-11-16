import sys
import jieba
import pinyin_predictor as ppredictor
import ngram_utilities as nutil

train_ngram = nutil.load_ngram(sys.argv[1])
train_ngram.load_smoothed_dict(sys.argv[2])
predictor = ppredictor.generate_predictor_instance(sys.argv[1],
                sys.argv[2], sys.argv[3])
print "please input"
while 1:
    line = sys.stdin.readline().rstrip('\n')
    cutted = list(jieba.cut(line))
    for s in cutted:
        print s
    print train_ngram.get_prob('|'.join(cutted))
    words, score = train_ngram.predict(cutted)
    print "cutted score ", score
    pruned = predictor.prune(cutted)
    print "pruned ? ", pruned
