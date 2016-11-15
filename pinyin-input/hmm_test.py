import sys
import hmm

def predict(hmm):
    print "predicting mode"
    while 1:
        line = sys.stdin.readline().rstrip('\n')
        print "predict ", line
        path = hmm.predict(line.split())
        for c in path:
           print c.decode('utf-8')
 
def debug(hmm):
    print "please input"
    while 1:
        line = sys.stdin.readline().rstrip('\n')
        print line
        fields = line.split()
        if len(fields) == 1:
            print hmm.get_initial_prob(fields[0])
            print hmm.get_trans_prob_vector(fields[0])

        elif len(fields) == 2:
            print hmm.get_trans_prob(fields[0], fields[1])

hidden_states, items, trans_prob, emission_prob,initial_prob = \
    hmm.load_data(sys.argv[1])
myhmm = MyHMM(hidden_states, items, trans_prob, emission_prob, initial_prob)

#line = "ni hao ba"
#line = "dry"
#print hmm.backward_compute(seq)
#print hmm.forward_compute(seq)
#hmm.compute_gamma()
#hmm.compute_Xi()

#predict(line, hmm)
if sys.argv[2] == 'debug':
    debug(myhmm)
else:
    predict(myhmm)
