import sys
import numpy as np

class MyHMM:
    START = -1
    END = -1
    
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
        for s, prob in enumerate(self.prob_matrix[seq_len-1]):
            if cur_max < prob:
                max_path[seq_len] = s
                cur_max = prob
                last_state = s
        for pos in range(seq_len - 1, 0, -1):
            print pos
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
            print self.prob_matrix
            return
        self.propagate(pos-1)
        for s2 in range(self.num_hidden_states):
            cur_max = -1
            for s1 in range(self.num_hidden_states):
                prob = self.prob_matrix[pos-1][s1] \
                    * self.trans_matrix[s1][s2] \
                    * self.emission_matrix[s2][item]
                if cur_max < prob:
                    self.path[pos][s2] = s1
                    cur_max = prob
                    self.prob_matrix[pos][s2] = cur_max = prob
        print self.prob_matrix
        print self.path


    def forward_compute(self, seq, status):
        seq_len = len(obs_seq)
        if seq_len == 0: return
        self.obs_seq = self.convert_item_seq(obs_seq)
        self.prob_matrix = np.ndarray(shape=(seq_len+1, self.num_hidden_states+1),
                dtype=float, order='C')
        self.prob_matrix.fill(0)
        return self.forward(seq_len+1, self.END)

 
    def forward(self, pos, status):
        item = self.obs_items[pos]
        total_prob = 0
        if pos == 0:
            total_prob = self.initial_prob[status] * self.emission_prob[status][item]
        elif status == self.END
            total_prob = 1
        else
            for prev_s in self.hidden_states:
                prev_prob = self.forward(pos-1, prev_s)
                if status == self.END:
                    cur_prob = prev_prob
                else:
                    cur_prob = prev_prob * \
                        self.trans_matrix[prev_s][status] * \
                        emission_prob
                total_prob += cur_prob
        self.prob_matrix[pos][status] = total_prob
        return total_prob
                `

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
    print hidden_states
    print items
    print trans_prob
    print emission_prob
    print initial_prob
    return (hidden_states, items, trans_prob, emission_prob, initial_prob)


hidden_states, items, trans_prob, emission_prob,initial_prob = \
    load_data(sys.argv[1])
hmm = MyHMM(hidden_states, items, trans_prob, emission_prob, initial_prob)

line = "1 1 1 1 2 1 2 2 2 2"
print hmm.forward_compute(line.split())
#while 1:
#    line = sys.stdin.readline()
#path = hmm.predict(line.split())
print path
