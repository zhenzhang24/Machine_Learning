import sys
import numpy as np
import hmm

class MyHMM:
    START = -1
    END = -2
    
    def __init__(self,
            hidden_states, 
            obs_items, 
            trans_matrix, 
            emission_matrix,
            initial_prob):
        self.states_order_map = {'START':-1}
        self.items_order_map = {}
        self.order_states_map = {-1:'START'}
        self.order_items_map = {}
        for i, s in enumerate(hidden_states):
            self.states_order_map[s] = i
            self.order_states_map[i] = s
        for i, item in enumerate(obs_items):
            self.items_order_map[item] = i
            self.order_items_map[i] = item
        self.hidden_states = self.convert_states(hidden_states)
        self.obs_items = self.convert_item_seq(obs_items)
        self.trans_matrix = trans_matrix
        self.initial_prob = initial_prob
        self.emission_matrix = emission_matrix
        self.num_hidden_states = len(self.hidden_states)
        self.num_obs_items = len(self.obs_items)

    def convert_item_seq(self, obs_seq):
        return map(lambda x: self.items_order_map[x], obs_seq)

    def convert_states(self, states):
        return map(lambda x: self.states_order_map[x], states)

    def get_trans_prob(self, s1, s2):
        if self.states_order_map.has_key(s1) and self.states_order_map.has_key(s2):
            return self.trans_matrix[self.states_order_map[s1]][self.states_order_map[s2]]
        else:
            return 0

    def get_initial_prob(self, s):
        if self.initial_prob.has_key(s):
            return self.initial_prob[self.states_order_map[s]]
        else:
            return 0

    def get_trans_prob_vector(self, s):
        if self.states_order_map.has_key(s):
            return self.trans_matrix[self.states_order_map[s]]
        else:
            return 0

    def trace(self):
        print self.prob_matrix
        for i in range(len(self.prob_matrix)):
            print 'state %d %s' % (i, self.order_states_map[i])
            for s, prob in enumerate(self.prob_matrix[i]):
                if prob > 0:
                    print '%s:%f' % (self.order_states_map[s], prob)
            print '----------------------------------------------------'

    def predict(self, obs_seq):
        seq_len = len(obs_seq)
        if seq_len == 0: return
        self.obs_seq = self.convert_item_seq(obs_seq)
        self.prob_matrix = np.ndarray(shape=(seq_len, self.num_hidden_states),
                dtype=float, order='C')
        self.prob_matrix.fill(0)
        self.path = np.ndarray(shape=(seq_len, self.num_hidden_states),
                dtype=int, order='C')
        self.path.fill(-1)
        self.propagate(seq_len-1)

        # backward to find the path
        max_path = [-1] * (seq_len + 1)
        cur_max = -1
        last_state = -1
        #self.trace()
        for s, prob in enumerate(self.prob_matrix[seq_len-1]):
            if cur_max < prob:
                max_path[seq_len] = s
                cur_max = prob
                last_state = s
        for pos in range(seq_len - 1, 0, -1):
            max_path[pos] = self.path[pos][last_state]
            last_state = max_path[pos]

        return map(lambda x : self.order_states_map[x], max_path)

    def propagate(self, pos):
        cur_max = -1
        item = self.obs_seq[pos]
        if pos == 0:
            for s in range(self.num_hidden_states):
                prob = self.prob_matrix[0][s] = \
                        self.emission_matrix[s][item] * self.initial_prob[s]
                self.path[0][s] = s
            return
        self.propagate(pos-1)
        for s2 in range(self.num_hidden_states):
            #-----------------
            if self.emission_matrix[s2][item] == 0:
                self.path[pos][s2] = 0
                self.prob_matrix[pos][s2] = 0
                continue
            #------------------
            cur_max = -1
            for s1 in range(self.num_hidden_states):
                prob = self.prob_matrix[pos-1][s1] \
                    * self.trans_matrix[s1][s2] \
                    * self.emission_matrix[s2][item]
                if cur_max < prob:
                    self.path[pos][s2] = s1
                    cur_max = prob
                    self.prob_matrix[pos][s2] = cur_max = prob
        #print self.prob_matrix
        #print self.path


    def forward_compute(self, obs_seq):
        seq_len = len(obs_seq)
        if seq_len == 0: return
        self.obs_seq = self.convert_item_seq(obs_seq)
        self.forward_prob_matrix = np.ndarray(shape=(seq_len+1, self.num_hidden_states+1),
                dtype=float, order='C')
        self.forward_prob_matrix.fill(0)
        self.max_prob_for_seq = self.forward(seq_len, self.END)
        print self.forward_prob_matrix
        return  self.max_prob_for_seq 

    # forward prob A(i, t) = P(O1, .., Ot, Si=Qt | model) 
    # given a model
    # it represent the probability that we observe the previous
    # seq till time t and we are in state Si at time t
    def forward(self, pos, status):
        total_prob = 0
        if pos == 0:
            item = self.obs_seq[pos]
            total_prob = self.initial_prob[status]\
                    * self.emission_matrix[status][item]
        else:
            for prev_s in self.hidden_states:
                if self.forward_prob_matrix[pos-1][prev_s] != 0:
                    prev_prob = self.forward_prob_matrix[pos-1][prev_s]
                else:
                    prev_prob = self.forward(pos-1, prev_s)
                if status == self.END:
                    cur_prob = prev_prob
                else:
                    item = self.obs_seq[pos]
                    cur_prob = prev_prob * \
                        self.trans_matrix[prev_s][status] * \
                        self.emission_matrix[status][item]
                total_prob += cur_prob
        self.forward_prob_matrix[pos][status] = total_prob
        return total_prob

    def backward_compute(self, obs_seq):
        seq_len = len(obs_seq)
        if seq_len == 0: return
        self.obs_seq = self.convert_item_seq(obs_seq)
        self.backward_prob_matrix = np.ndarray(shape=(seq_len+1, self.num_hidden_states+1),
                dtype=float, order='C')
        self.backward_prob_matrix.fill(0)
        prob = self.backward(-1, self.START)
        print self.backward_prob_matrix
        return prob

    # backward prob B(i, t) = P(Ot+1, ..., OT | Si = Qt, model)
    # given model and at time t we are in state i and emit Ot
    # it represents the probability that we observe the rest the seq
    def backward(self, pos, status):
        total_prob = 0
        if pos == len(self.obs_seq)-1:
            self.backward_prob_matrix[pos][status] = 1
            return 1
        for next_s in self.hidden_states:
            if self.backward_prob_matrix[pos+1][next_s] != 0:
                next_prob = self.backward_prob_matrix[pos+1][next_s]
            else:
                next_prob = self.backward(pos+1, next_s)
            if status == self.START:
                trans_prob = self.initial_prob[next_s]
            else:
                trans_prob = self.trans_matrix[status][next_s]

            item = self.obs_seq[pos+1]
            cur_prob = next_prob * \
                    trans_prob * \
                    self.emission_matrix[next_s][item]
            total_prob += cur_prob
        self.backward_prob_matrix[pos][status] = total_prob
        return total_prob

    # gamma(i, t) = P(Qt=Si | O, model)
    # given observation and model
    # it presents the probability that at time t, we are in state i
    def compute_gamma(self):
        seq_len = len(self.obs_seq)
        self.gamma_matrix = np.ndarray(shape=(seq_len, self.num_hidden_states),
                dtype=float, order='C')
        self.gamma_matrix.fill(0)
        for pos in range(seq_len):
            for status in self.hidden_states:
                self.gamma_matrix[pos][status] = \
                    self.forward_prob_matrix[pos][status] * \
                    self.backward_prob_matrix[pos][status] / \
                    self.max_prob_for_seq
        print self.gamma_matrix

    # Xi(t, i, j) = P(Qt=Si, Qt+1 = Sj | O, model)
    # givne observation and model, 
    # it represents the probability that 
    # a transition from i to j happens at time t
    def compute_Xi(self):
        seq_len = len(self.obs_seq)
        self.Xi = np.ndarray(
                shape=(seq_len, self.num_hidden_states, self.num_hidden_states),
                dtype=float,
                order='C')
        self.Xi.fill(0)
        for pos in range(seq_len-1):
            item = self.obs_seq[pos+1]
            for i in self.hidden_states:
                for j in self.hidden_states:
                    self.Xi[pos][i][j] = \
                            self.forward_prob_matrix[pos][i] * \
                            self.backward_prob_matrix[pos+1][j] * \
                            self.emission_matrix[j][item] * \
                            self.trans_matrix[i][j] / self.max_prob_for_seq

        print self.Xi
#
def load_data(trans_file):
    f = open(trans_file)
    read_status = None
    hidden_states = []
    items = []
    trans_array = []
    item_array = []
    initial_prob = []
    num_hidden_states = 0
    num_items = 0

    for line in f.readlines():
        line = line.strip()
        if line == None:
            continue
        fields = line.split()
        if line == "hidden states:":
            read_status = "states"
            continue
        if read_status == "states":
            hidden_states = fields
            num_hidden_states = len(hidden_states)
            read_status = None
        if line == "items:":
            read_status = "items"
            continue
        if read_status == "items":
            items = fields
            num_items = len(items)
            read_status = None
        if line == "trans prob:":
            read_status = "trans"
            trans_array = []
            continue
        if read_status == "trans" and line != "end trans prob":
            trans_array.append(map(lambda x: float(x), fields))
        if line == "end trans prob":
            trans_prob = np.ndarray(\
                    shape=(num_hidden_states, num_hidden_states),\
                    buffer=np.array(trans_array),\
                    dtype=float,\
                    order = 'C')
            read_status = None
        if line == "emission prob:":
            read_status = "emission"
            item_array = []
            continue
        if read_status == "emission" and line != "end emission prob":
            item_array.append(map(lambda x: float(x), fields))
        if line == "end emission prob":
            emission_prob = np.ndarray(\
                    shape=(num_hidden_states, num_items),\
                    buffer=np.array(item_array),\
                    dtype=float,\
                    order = 'C')
            read_status = None
        if line == "initial prob:":
            read_status = "initial"
            continue
        if read_status == "initial":
            initial_prob = map(lambda x: float(x), fields)
            read_status = None
    f.close()
    #print hidden_states
    #print items
    #print trans_prob
    #print emission_prob
    #print initial_prob
    return (hidden_states, items, trans_prob, emission_prob, initial_prob)
