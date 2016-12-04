import sys
import lr
import lr_utils as utils


my_lr = lr.MyLogisticRegression(iter=int(sys.argv[2]))
lr.g_debug = int(sys.argv[3])
res = my_lr.train(sys.argv[1])
accuracy = utils.calc_accuracy(res, my_lr.target)
print "FINAl RESULTS"
print res
print "accuracy: %f" % accuracy
