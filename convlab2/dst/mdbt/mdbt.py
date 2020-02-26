import copy
import json
import os

import tensorflow as tf

from convlab2.dst.mdbt.mdbt_util import model_definition, \
    track_dialogue, generate_batch, process_history
from convlab2.dst.rule.multiwoz import normalize_value
from convlab2.util.multiwoz.state import default_state
from convlab2.dst.dst import DST
from convlab2.util.multiwoz.multiwoz_slot_trans import REF_SYS_DA, REF_USR_DA

from os.path import dirname

train_batch_size = 1
batches_per_eval = 10
no_epochs = 600
device = "gpu"
start_batch = 0


class MDBT(DST):
    """
    A multi-domain belief tracker, adopted from https://github.com/osmanio2/multi-domain-belief-tracking.
    """
    def __init__(self, ontology_vectors, ontology, slots, data_dir):
        DST.__init__(self)
        # data profile
        self.data_dir = data_dir
        self.validation_url = os.path.join(self.data_dir, 'data/validate.json')
        self.word_vectors_url = os.path.join(self.data_dir, 'word-vectors/paragram_300_sl999.txt')
        self.training_url = os.path.join(self.data_dir, 'data/train.json')
        self.ontology_url = os.path.join(self.data_dir, 'data/ontology.json')
        self.testing_url = os.path.join(self.data_dir, 'data/test.json')
        self.model_url = os.path.join(self.data_dir, 'models/model-1')
        self.graph_url = os.path.join(self.data_dir, 'graphs/graph-1')
        self.results_url = os.path.join(self.data_dir, 'results/log-1.txt')
        self.kb_url = os.path.join(self.data_dir, 'data/')  # not used
        self.train_model_url = os.path.join(self.data_dir, 'train_models/model-1')
        self.train_graph_url = os.path.join(self.data_dir, 'train_graph/graph-1')

        self.model_variables = model_definition(ontology_vectors, len(ontology), slots, num_hidden=None,
                                                bidir=True, net_type=None, test=True, dev='cpu')
        self.state = default_state()
        _config = tf.ConfigProto()
        _config.gpu_options.allow_growth = True
        _config.allow_soft_placement = True
        self.sess = tf.Session(config=_config)
        self.param_restored = False
        self.det_dic = {}
        for domain, dic in REF_USR_DA.items():
            for key, value in dic.items():
                assert '-' not in key
                self.det_dic[key.lower()] = key + '-' + domain
                self.det_dic[value.lower()] = key + '-' + domain

        def parent_dir(path, time=1):
            for _ in range(time):
                path = os.path.dirname(path)
            return path
        root_dir = parent_dir(os.path.abspath(__file__), 4)
        self.value_dict = json.load(open(os.path.join(root_dir, 'data/multiwoz/value_dict.json')))

    def init_session(self):
        self.state = default_state()
        if not self.param_restored:
            self.restore()

    def restore(self):
        self.__restore_model(self.sess, tf.train.Saver())

    def update_batch(self, batch_action):
        pass

    def update(self, user_act=None):
        """Update the dialog state."""
        if type(user_act) is not str:
            raise Exception('Expected user_act to be <class \'str\'> type, but get {}.'.format(type(user_act)))
        prev_state = self.state
        if not os.path.exists(os.path.join(self.data_dir, "results")):
            os.makedirs(os.path.join(self.data_dir, "results"))

        global train_batch_size

        model_variables = self.model_variables
        (user, sys_res, no_turns, user_uttr_len, sys_uttr_len, labels, domain_labels, domain_accuracy,
         slot_accuracy, value_accuracy, value_f1, train_step, keep_prob, predictions,
         true_predictions, [y, _]) = model_variables

        # generate fake dialogue based on history (this os to reuse the original MDBT code)
        # actual_history = prev_state['history']  # [[sys, user], [sys, user], ...]
        actual_history = copy.deepcopy(prev_state['history'])  # [[sys, user], [sys, user], ...]
        actual_history = [['null']]
        actual_history[-1].append(user_act)
        actual_history = self.normalize_history(actual_history)
        if len(actual_history) == 0:
            actual_history = [['', user_act if len(user_act)>0 else 'fake user act']]
        fake_dialogue = {}
        turn_no = 0
        for _sys, _user in actual_history:
            turn = {}
            turn['system'] = _sys
            fake_user = {}
            fake_user['text'] = _user
            fake_user['belief_state'] = default_state()['belief_state']
            turn['user'] = fake_user
            key = str(turn_no)
            fake_dialogue[key] = turn
            turn_no += 1
        context, actual_context = process_history([fake_dialogue], self.word_vectors, self.ontology)
        batch_user, batch_sys, batch_labels, batch_domain_labels, batch_user_uttr_len, batch_sys_uttr_len, \
                batch_no_turns = generate_batch(context, 0, 1, len(self.ontology))  # old feature

        # run model
        [pred, y_pred] = self.sess.run(
            [predictions, y],
            feed_dict={user: batch_user, sys_res: batch_sys,
                       labels: batch_labels,
                       domain_labels: batch_domain_labels,
                       user_uttr_len: batch_user_uttr_len,
                       sys_uttr_len: batch_sys_uttr_len,
                       no_turns: batch_no_turns,
                       keep_prob: 1.0})

        # convert to str output
        dialgs, _, _ = track_dialogue(actual_context, self.ontology, pred, y_pred)
        assert len(dialgs) >= 1
        last_turn = dialgs[0][-1]
        predictions = last_turn['prediction']
        new_belief_state = copy.deepcopy(prev_state['belief_state'])

        # updaet belief state
        for item in predictions:
            item = item.lower()
            domain, slot, value = item.strip().split('-')
            value = value[::-1].split(':', 1)[1][::-1]
            if slot == 'price range':
                slot = 'pricerange'
            if slot not in ['name', 'book']:
                if domain not in new_belief_state:
                    raise Exception('Error: domain <{}> not in belief state'.format(domain))
                slot = REF_SYS_DA[domain.capitalize( )].get(slot, slot)
                assert 'semi' in new_belief_state[domain]
                assert 'book' in new_belief_state[domain]
                if 'book' in slot:
                    assert slot.startswith('book ')
                    slot = slot.strip().split()[1]
                if slot == 'arriveby':
                    slot = 'arriveBy'
                elif slot == 'leaveat':
                    slot = 'leaveAt'
                domain_dic = new_belief_state[domain]
                if slot in domain_dic['semi']:
                    new_belief_state[domain]['semi'][slot] = normalize_value(self.value_dict, domain, slot, value)
                elif slot in domain_dic['book']:
                    new_belief_state[domain]['book'][slot] = value
                elif slot.lower() in domain_dic['book']:
                    new_belief_state[domain]['book'][slot.lower()] = value
                else:
                    with open('mdbt_unknown_slot.log', 'a+') as f:
                        f.write('unknown slot name <{}> with value <{}> of domain <{}>\nitem: {}\n\n'.format(slot, value,
                                domain, item))
        new_request_state = copy.deepcopy(prev_state['request_state'])
        # update request_state
        user_request_slot = self.detect_requestable_slots(user_act)
        for domain in user_request_slot:
            for key in user_request_slot[domain]:
                if domain not in new_request_state:
                    new_request_state[domain] = {}
                if key not in new_request_state[domain]:
                    new_request_state[domain][key] = user_request_slot[domain][key]
        # update state
        new_state = copy.deepcopy(dict(prev_state))
        new_state['belief_state'] = new_belief_state
        new_state['request_state'] = new_request_state
        self.state = new_state
        return self.state

    def normalize_history(self, history):
        """Replace zero-length history."""
        for i in range(len(history)):
            a, b = history[i]
            if len(a) == 0:
                history[i][0] = 'sys'
            if len(b) == 0:
                history[i][1] = 'user'
        return history

    def detect_requestable_slots(self, observation):
        result = {}
        observation = observation.lower()
        _observation = ' {} '.format(observation)
        for value in self.det_dic.keys():
            _value = ' {} '.format(value.strip())
            if _value in _observation:
                key, domain = self.det_dic[value].split('-')
                if domain not in result:
                    result[domain] = {}
                result[domain][key] = 0
        return result

    def __restore_model(self, sess, saver):
        saver.restore(sess, self.model_url)
        print('Loading trained MDBT model from ', self.model_url)
        self.param_restored = True

