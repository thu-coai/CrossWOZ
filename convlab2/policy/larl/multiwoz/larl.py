# Modified by Microsoft Corporation.
# Licensed under the MIT license.
import sys
#sys.path.append("/home/mawenchang/Convlab-2/")
from convlab2.policy.larl.multiwoz.corpora_inference import BOS, EOS, PAD
from convlab2.policy.larl.multiwoz.latent_dialog.enc2dec.decoders import DecoderRNN
from convlab2.policy.larl.multiwoz.latent_dialog.utils import INT, FLOAT, LONG, Pack, cast_type
from convlab2.policy.larl.multiwoz.latent_dialog.utils import get_detokenize
from convlab2.policy.larl.multiwoz.utils.nlp import normalize
from convlab2.policy.larl.multiwoz.utils import util, delexicalize
from convlab2.policy.larl.multiwoz import corpora_inference
from convlab2.policy.larl.multiwoz.latent_dialog import domain
from convlab2.policy.larl.multiwoz.latent_dialog.models_task import SysPerfectBD2Cat
from convlab2.policy import Policy
from convlab2.util.file_util import cached_path
from convlab2.util.multiwoz.state import default_state
from convlab2.util.multiwoz.dbquery import Database
from copy import deepcopy
import json
import os
import random
import tempfile
import zipfile

import numpy as np
import re
import torch
from nltk import word_tokenize
from torch.autograd import Variable
import pickle


TEACH_FORCE = 'teacher_forcing'
TEACH_GEN = 'teacher_gen'
GEN = 'gen'
GEN_VALID = 'gen_valid'




def oneHotVector(num, domain, vector):
    """Return number of available entities for particular domain."""
    domains = ['restaurant', 'hotel', 'attraction', 'train']
    number_of_options = 6
    if domain != 'train':
        idx = domains.index(domain)
        if num == 0:
            vector[idx * 6: idx * 6 + 6] = np.array([1, 0, 0, 0, 0, 0])
        elif num == 1:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 1, 0, 0, 0, 0])
        elif num == 2:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 1, 0, 0, 0])
        elif num == 3:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 1, 0, 0])
        elif num == 4:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 0, 1, 0])
        elif num >= 5:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 0, 0, 1])
    else:
        idx = domains.index(domain)
        if num == 0:
            vector[idx * 6: idx * 6 + 6] = np.array([1, 0, 0, 0, 0, 0])
        elif num <= 2:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 1, 0, 0, 0, 0])
        elif num <= 5:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 1, 0, 0, 0])
        elif num <= 10:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 1, 0, 0])
        elif num <= 40:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 0, 1, 0])
        elif num > 40:
            vector[idx * 6: idx * 6 + 6] = np.array([0, 0, 0, 0, 0, 1])

    return vector


def addBookingPointer(state, pointer_vector):
    """Add information about availability of the booking option."""
    # Booking pointer
    rest_vec = np.array([1, 0])
    if "book" in state['restaurant']:
        if "booked" in state['restaurant']['book']:
            if state['restaurant']['book']["booked"]:
                if "reference" in state['restaurant']['book']["booked"][0]:
                    rest_vec = np.array([0, 1])

    hotel_vec = np.array([1, 0])
    if "book" in state['hotel']:
        if "booked" in state['hotel']['book']:
            if state['hotel']['book']["booked"]:
                if "reference" in state['hotel']['book']["booked"][0]:
                    hotel_vec = np.array([0, 1])

    train_vec = np.array([1, 0])
    if "book" in state['train']:
        if "booked" in state['train']['book']:
            if state['train']['book']["booked"]:
                if "reference" in state['train']['book']["booked"][0]:
                    train_vec = np.array([0, 1])

    pointer_vector = np.append(pointer_vector, rest_vec)
    pointer_vector = np.append(pointer_vector, hotel_vec)
    pointer_vector = np.append(pointer_vector, train_vec)

    # pprint(pointer_vector)
    return pointer_vector


def addDBPointer(state,db):
    """Create database pointer for all related domains."""
    domains = ['restaurant', 'hotel', 'attraction', 'train']
    pointer_vector = np.zeros(6 * len(domains))
    db_results = {}
    num_entities = {}
    for domain in domains:
        # entities = dbPointer.queryResultVenues(domain, {'metadata': state})
        entities = db.query(domain, state[domain]['semi'].items())
        num_entities[domain] = len(entities)
        if len(entities) > 0:
            # fields = dbPointer.table_schema(domain)
            # db_results[domain] = dict(zip(fields, entities[0]))
            db_results[domain] = entities[0]
        # pointer_vector = dbPointer.oneHotVector(len(entities), domain, pointer_vector)
        pointer_vector = oneHotVector(len(entities), domain, pointer_vector)

    return list(pointer_vector), db_results, num_entities


def delexicaliseReferenceNumber(sent, state):
    """Based on the belief state, we can find reference number that
    during data gathering was created randomly."""
    domains = ['restaurant', 'hotel', 'attraction',
               'train', 'taxi', 'hospital']  # , 'police']
    for domain in domains:
        if state[domain]['book']['booked']:
            for slot in state[domain]['book']['booked'][0]:
                if slot == 'reference':
                    val = '[' + domain + '_' + slot + ']'
                else:
                    val = '[' + domain + '_' + slot + ']'
                key = normalize(state[domain]['book']['booked'][0][slot])
                sent = (' ' + sent + ' ').replace(' ' +
                                                  key + ' ', ' ' + val + ' ')

                # try reference with hashtag
                key = normalize("#" + state[domain]['book']['booked'][0][slot])
                sent = (' ' + sent + ' ').replace(' ' +
                                                  key + ' ', ' ' + val + ' ')

                # try reference with ref#
                key = normalize(
                    "ref#" + state[domain]['book']['booked'][0][slot])
                sent = (' ' + sent + ' ').replace(' ' +
                                                  key + ' ', ' ' + val + ' ')
    return sent


def domain_mark_not_mentioned(state, active_domain):
    if domain not in ['police', 'hospital', 'taxi', 'train', 'attraction', 'restaurant', 'hotel']:
        pass

    for s in state[active_domain]['semi']:
        if state[active_domain]['semi'][s] == '':
            state[active_domain]['semi'][s] = 'not mentioned'


def mark_not_mentioned(state):
    for domain in state:
        # if domain == 'history':
        if domain not in ['police', 'hospital', 'taxi', 'train', 'attraction', 'restaurant', 'hotel']:
            continue
        try:
            # if len([s for s in state[domain]['semi'] if s != 'book' and state[domain]['semi'][s] != '']) > 0:
                # for s in state[domain]['semi']:
                #     if s != 'book' and state[domain]['semi'][s] == '':
                #         state[domain]['semi'][s] = 'not mentioned'
            for s in state[domain]['semi']:
                if state[domain]['semi'][s] == '':
                    state[domain]['semi'][s] = 'not mentioned'
        except Exception as e:
            # print(str(e))
            # pprint(state[domain])
            pass


def get_summary_bstate(bstate):
    """Based on the mturk annotations we form multi-domain belief state"""
    domains = [u'taxi', u'restaurant',  u'hospital',
               u'hotel', u'attraction', u'train', u'police']
    summary_bstate = []
    for domain in domains:
        domain_active = False

        booking = []
        for slot in sorted(bstate[domain]['book'].keys()):
            if slot == 'booked':
                if bstate[domain]['book']['booked']:
                    booking.append(1)
                else:
                    booking.append(0)
            else:
                if bstate[domain]['book'][slot] != "":
                    booking.append(1)
                else:
                    booking.append(0)
        if domain == 'train':
            if 'people' not in bstate[domain]['book'].keys():
                booking.append(0)
            if 'ticket' not in bstate[domain]['book'].keys():
                booking.append(0)
        summary_bstate += booking

        for slot in bstate[domain]['semi']:
            slot_enc = [0, 0, 0]
            if bstate[domain]['semi'][slot] == 'not mentioned':
                slot_enc[0] = 1
            elif bstate[domain]['semi'][slot] == 'dont care' or bstate[domain]['semi'][slot] == 'dontcare' or bstate[domain]['semi'][slot] == "don't care":
                slot_enc[1] = 1
            elif bstate[domain]['semi'][slot]:
                slot_enc[2] = 1
            if slot_enc != [0, 0, 0]:
                domain_active = True
            summary_bstate += slot_enc

        # quasi domain-tracker
        if domain_active:
            summary_bstate += [1]
        else:
            summary_bstate += [0]

    # print(len(summary_bstate))
    assert len(summary_bstate) == 94
    return summary_bstate


DEFAULT_CUDA_DEVICE = -1
DEFAULT_DIRECTORY = "models"
DEFAULT_ARCHIVE_FILE = os.path.join(DEFAULT_DIRECTORY, "milu.tar.gz")


class LaRL(Policy):
    def __init__(self,
                 archive_file=DEFAULT_ARCHIVE_FILE,
                 cuda_device=DEFAULT_CUDA_DEVICE,
                 model_file=None):

        if not os.path.isfile(archive_file):
            if not model_file:
                raise Exception("No model for LaRL is specified!")
            archive_file = cached_path(model_file)


        temp_path = tempfile.mkdtemp()
        print(temp_path)
        zip_ref = zipfile.ZipFile(archive_file, 'r')
        zip_ref.extractall(temp_path)
        zip_ref.close()

        self.prev_state = default_state()
        self.prev_active_domain = None

        domain_name = 'object_division'
        domain_info = domain.get_domain(domain_name)
        self.db=Database()
        data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
        train_data_path = os.path.join(data_path, 'norm-multi-woz', 'train_dials.json')
        if not os.path.exists(train_data_path):
            zipped_file = os.path.join(data_path, 'norm-multi-woz.zip')
            archive = zipfile.ZipFile(zipped_file, 'r')
            archive.extractall(data_path)

        norm_multiwoz_path = os.path.join(data_path, 'norm-multi-woz')
        with open(os.path.join(norm_multiwoz_path, 'input_lang.index2word.json')) as f:
            self.input_lang_index2word = json.load(f)
        with open(os.path.join(norm_multiwoz_path, 'input_lang.word2index.json')) as f:
            self.input_lang_word2index = json.load(f)
        with open(os.path.join(norm_multiwoz_path, 'output_lang.index2word.json')) as f:
            self.output_lang_index2word = json.load(f)
        with open(os.path.join(norm_multiwoz_path, 'output_lang.word2index.json')) as f:
            self.output_lang_word2index = json.load(f)

        config = Pack(
            seed=10,
            train_path=train_data_path,
            max_vocab_size=1000,
            last_n_model=5,
            max_utt_len=50,
            max_dec_len=50,
            backward_size=2,
            batch_size=1,
            use_gpu=True,
            op='adam',
            init_lr=0.001,
            l2_norm=1e-05,
            momentum=0.0,
            grad_clip=5.0,
            dropout=0.5,
            max_epoch=100,
            embed_size=100,
            num_layers=1,
            utt_rnn_cell='gru',
            utt_cell_size=300,
            bi_utt_cell=True,
            enc_use_attn=True,
            dec_use_attn=True,
            dec_rnn_cell='lstm',
            dec_cell_size=300,
            dec_attn_mode='cat',
            y_size=10,
            k_size=20,
            beta=0.001,
            simple_posterior=True,
            contextual_posterior=True,
            use_mi=False,
            use_pr=True,
            use_diversity=False,
            #
            beam_size=20,
            fix_batch=True,
            fix_train_batch=False,
            avg_type='word',
            print_step=300,
            ckpt_step=1416,
            improve_threshold=0.996,
            patient_increase=2.0,
            save_model=True,
            early_stop=False,
            gen_type='greedy',
            preview_batch_num=None,
            k=domain_info.input_length(),
            init_range=0.1,
            pretrain_folder='2019-09-20-21-43-06-sl_cat',
            forward_only=False
        )

        config.use_gpu = config.use_gpu and torch.cuda.is_available()
        self.corpus = corpora_inference.NormMultiWozCorpus(config)
        self.model = SysPerfectBD2Cat(self.corpus, config)
        self.config = config
        if config.use_gpu:
            self.model.load_state_dict(torch.load(
                os.path.join(temp_path, 'larl_model/best-model')))
            self.model.cuda()
        else:
            self.model.load_state_dict(torch.load(os.path.join(
                temp_path, 'larl_model/best-model'), map_location=lambda storage, loc: storage))
        self.model.eval()
        self.dic = pickle.load(
            open(os.path.join(temp_path, 'larl_model/svdic.pkl'), 'rb'))


    def reset():
        self.prev_state = default_state()
        
    def input_index2word(self, index):
        # if self.input_lang_index2word.has_key(index):
        if index in self.input_lang_index2word:
            return self.input_lang_index2word[index]
        else:
            raise UserWarning('We are using UNK')

    def output_index2word(self, index):
        # if self.output_lang_index2word.has_key(index):
        if index in self.output_lang_index2word:
            return self.output_lang_index2word[index]
        else:
            raise UserWarning('We are using UNK')

    def input_word2index(self, index):
        # if self.input_lang_word2index.has_key(index):
        if index in self.input_lang_word2index:
            return self.input_lang_word2index[index]
        else:
            return 2

    def output_word2index(self, index):
        # if self.output_lang_word2index.has_key(index):
        if index in self.output_lang_word2index:
            return self.output_lang_word2index[index]
        else:
            return 2

    def np2var(self, inputs, dtype):
        if inputs is None:
            return None
        return cast_type(Variable(torch.from_numpy(inputs)),
                         dtype,
                         self.config.use_gpu)

    def extract_short_ctx(self, context, context_lens, backward_size=1):
        utts = []
        for b_id in range(context.shape[0]):
            utts.append(context[b_id, context_lens[b_id]-1])
        return np.array(utts)

    def get_active_domain_test(self, prev_active_domain, prev_action, action):
        domains = ['hotel', 'restaurant', 'attraction',
                   'train', 'taxi', 'hospital', 'police']
        active_domain = None
        cur_action_keys=action.keys()
        state=[]
        for act in cur_action_keys:
          slots=act.split('-')
          action=slots[0].lower()
          state.append(action)


        #  print('get_active_domain')
        # for domain in domains:
        """for domain in range(len(domains)):
            domain = domains[i]
            if domain not in prev_state and domain not in state:
                continue
            if domain in prev_state and domain not in state:
                return domain
            elif domain not in prev_state and domain in state:
                return domain
            elif prev_state[domain] != state[domain]:
                active_domain = domain
        if active_domain is None:
            active_domain = prev_active_domain"""
        if len(state)!=0:
          active_domain=state[0]
        if active_domain is None:
          active_domain = prev_active_domain
        elif active_domain=="general":
          active_domain = prev_active_domain
        return active_domain


    def get_active_domain(self, prev_active_domain, prev_state, state):
        domains = ['hotel', 'restaurant', 'attraction',
                   'train', 'taxi', 'hospital', 'police']
        active_domain = None
        #print("PREV_STATE:",prev_state)
        #print()
        #print("NEW_STATE",state)
        #print()
        for domain in domains:
            if domain not in prev_state and domain not in state:
                continue
            if domain in prev_state and domain not in state:
                print("case 1:",domain)
                return domain
            elif domain not in prev_state and domain in state:
                print("case 2:",domain)
                return domain
            elif prev_state[domain] != state[domain]:
                print("case 3:",domain)
                active_domain = domain
        if active_domain is None:
            active_domain = prev_active_domain
        return active_domain
    def predict(self, state):
        try:
            response, active_domain = self.predict_response(state)
        except Exception as e:
            print('Response generation error', e)
            response = 'What did you say?'
            active_domain = None
        self.prev_state = deepcopy(state)
        self.prev_active_domain = active_domain

        return response

    def predict_response(self, state):
        history = []
        #print(state)
        for i in range(len(state['history'])):
            for j in range(len(state['history'][i])):
                history.append(state['history'][i][j])

        e_idx = len(history)
        s_idx = max(0, e_idx - self.config.backward_size)
        context = []
        for turn in history[s_idx: e_idx]:
            # turn = pad_to(config.max_utt_len, turn, do_pad=False)
            context.append(turn)
            
        if len(state['history']) == 1:
            self.prev_state = default_state()

        prepared_data = {}
        prepared_data['context'] = []
        prepared_data['response'] = {}

        prev_action = deepcopy(self.prev_state['user_action'])
        prev_bstate = deepcopy(self.prev_state['belief_state'])
        state_history = state['history']
        action = deepcopy(state['user_action'])
        bstate = deepcopy(state['belief_state'])

        # mark_not_mentioned(prev_state)
        #active_domain = self.get_active_domain_convlab(self.prev_active_domain, prev_bstate, bstate)
        active_domain = self.get_active_domain(self.prev_active_domain, prev_bstate, bstate)
        print(active_domain)
        domain_mark_not_mentioned(bstate, active_domain)

        top_results, num_results = None, None
        for usr in context:

            words = usr.split()

            usr = delexicalize.delexicalise(' '.join(words), self.dic)

            # parsing reference number GIVEN belief state
            usr = delexicaliseReferenceNumber(usr, bstate)

            # changes to numbers only here
            digitpat = re.compile('\d+')
            usr = re.sub(digitpat, '[value_count]', usr)
            # add database pointer
            pointer_vector, top_results, num_results = addDBPointer(bstate,self.db)
            #print(top_results)
            # add booking pointer
            pointer_vector = addBookingPointer(bstate, pointer_vector)
            belief_summary = get_summary_bstate(bstate)

            usr_utt = [BOS] + usr.split() + [EOS]
            packed_val = {}
            packed_val['bs'] = belief_summary
            packed_val['db'] = pointer_vector
            packed_val['utt'] = self.corpus._sent2id(usr_utt)

            prepared_data['context'].append(packed_val)

        prepared_data['response']['bs'] = prepared_data['context'][-1]['bs']
        prepared_data['response']['db'] = prepared_data['context'][-1]['db']
        results = [Pack(context=prepared_data['context'],
                        response=prepared_data['response'])]

        data_feed = prepare_batch_gen(results, self.config)

        outputs = self.model_predict(data_feed)
        if active_domain is not None and active_domain in num_results:
            num_results = num_results[active_domain]
        else:
            num_results = 0

        if active_domain is not None and active_domain in top_results:
            top_results = {active_domain: top_results[active_domain]}
        else:
            top_results = {}

        print(top_results)

        state_with_history = deepcopy(bstate)
        state_with_history['history'] = deepcopy(state_history)

        response = self.populate_template(
            outputs, top_results, num_results, state_with_history)

        #import pprint
        #pprint.pprint("============")
        #pprint.pprint('usr:')
        #pprint.pprint(context[-1])
        #pprint.pprint(outputs)
        #pprint.pprint('agent:')
        #pprint.pprint(response)
        #pprint.pprint("============")

        return response, active_domain

    def populate_template(self, template, top_results, num_results, state):
        #print("template:",template)
        #print("top_results:",top_results)
        active_domain = None if len(
            top_results.keys()) == 0 else list(top_results.keys())[0]
        template = template.replace(
            'book [value_count] of them', 'book one of them')
        tokens = template.split()
        response = []
        for index, token in enumerate(tokens):
            if token.startswith('[') and (token.endswith(']') or token.endswith('].') or token.endswith('],')):
                domain = token[1:-1].split('_')[0]
                slot = token[1:-1].split('_')[1]
                if slot.endswith(']'):
                    slot = slot[:-1]
                if domain == 'train' and slot == 'id':
                    slot = 'trainID'
                elif domain != 'train' and slot == 'price':
                    slot = 'pricerange'
                elif slot == 'reference':
                    slot = 'Ref'
                if domain in top_results and len(top_results[domain]) > 0 and slot in top_results[domain]:
                    # print('{} -> {}'.format(token, top_results[domain][slot]))
                    response.append(top_results[domain][slot])
                elif domain == 'value':
                    if slot == 'count':
                        if index + 1 < len(tokens):
                            if 'minute' in tokens[index+1] and active_domain == 'train':
                                response.append(top_results['train']['duration'].split()[0])
                            elif 'star' in tokens[index+1] and active_domain == 'hotel':
                                response.append(top_results['hotel']['stars'])
                            else:
                                response.append(str(num_results))
                        else:
                            response.append(str(num_results))
                    elif slot == 'place':
                        if 'arrive' in response or 'to' in response or 'arriving' in response:
                            for d in state:
                                if d == 'history':
                                    continue
                                if 'destination' in state[d]['semi']:
                                    response.append(
                                        state[d]['semi']['destination'])
                                    break
                        elif 'leave' in response or 'leaving' in response:
                            for d in state:
                                if d == 'history':
                                    continue
                                if 'departure' in state[d]['semi']:
                                    response.append(
                                        state[d]['semi']['departure'])
                                    break
                        else:
                            try:
                                for d in state:
                                    if d == 'history':
                                        continue
                                    for s in ['destination', 'departure']:
                                        if s in state[d]['semi']:
                                            response.append(
                                                state[d]['semi'][s])
                                            raise
                            except:
                                pass
                            else:
                                response.append(token)
                    elif slot == 'time':
                        if 'arrive' in ' '.join(response[-3:]) or 'arrival' in ' '.join(response[-3:]) or 'arriving' in ' '.join(response[-3:]):
                            if active_domain is not None and 'arriveBy' in top_results[active_domain]:
                                # print('{} -> {}'.format(token, top_results[active_domain]['arriveBy']))
                                response.append(
                                    top_results[active_domain]['arriveBy'])
                                continue
                            for d in state:
                                if d == 'history':
                                    continue
                                if 'arriveBy' in state[d]['semi']:
                                    response.append(
                                        state[d]['semi']['arriveBy'])
                                    break
                        elif 'leave' in ' '.join(response[-3:]) or  'leaving' in ' '.join(response[-3:]) or 'departure' in ' '.join(response[-3:]):
                            if active_domain is not None and 'leaveAt' in top_results[active_domain]:
                                # print('{} -> {}'.format(token, top_results[active_domain]['leaveAt']))
                                response.append(
                                    top_results[active_domain]['leaveAt'])
                                continue
                            for d in state:
                                if d == 'history':
                                    continue
                                if 'leaveAt' in state[d]['semi']:
                                    response.append(
                                        state[d]['semi']['leaveAt'])
                                    break
                        elif 'book' in response:
                            if state['restaurant']['book']['time'] != "":
                                response.append(
                                    state['restaurant']['book']['time'])
                        else:
                            try:
                                for d in state:
                                    if d == 'history':
                                        continue
                                    for s in ['arriveBy', 'leaveAt']:
                                        if s in state[d]['semi']:
                                            response.append(
                                                state[d]['semi'][s])
                                            raise
                            except:
                                pass
                            else:
                                response.append(token)
                    elif slot == 'price' and active_domain == 'attraction':
                        value = top_results['attraction']['entrance fee'].split()[0]
                        try: 
                            value = str(int(value))
                        except:
                            value = 'free'
                        response.append(value)
                    else:
                        # slot-filling based on query results
                        for d in top_results:
                            if slot in top_results[d]:
                                response.append(top_results[d][slot])
                                break
                        else:
                            # slot-filling based on belief state
                            for d in state:
                                if d == 'history':
                                    continue
                                if slot in state[d]['semi']:
                                    response.append(state[d]['semi'][slot])
                                    break
                            else:
                                response.append(token)
                else:
                    if domain == 'hospital':
                        if slot == 'phone':
                            response.append('01223216297')
                        elif slot == 'department':
                            response.append('neurosciences critical care unit')
                        elif slot == 'address':
                            response.append("Lincoln street")
                    elif domain == 'police':
                        if slot == 'phone':
                            response.append('01223358966')
                        elif slot == 'name':
                            response.append('Parkside Police Station')
                        elif slot == 'address':
                            response.append('Parkside, Cambridge')
                        elif slot == 'postcode':
                            response.append('533420')
                    elif domain == 'taxi':
                        if slot == 'phone':
                            response.append('01223358966')
                        elif slot == 'color':
                            response.append('white')
                        elif slot == 'type':
                            response.append('toyota')
                    elif domain == 'hotel':
                        if slot == 'address':
                            response.append('Bond Street, London')
                        elif slot == 'name':
                            response.append('Warwick')
                    elif domain == 'restaurant':
                        if slot == 'phone':
                            response.append('01223358963')
                        elif slot == 'name':
                            response.append('Korean BBQ')        
                        elif slot == 'postcode':
                            response.append('533482')
                    else:
                        # print(token)
                        response.append(token)
            else:
                response.append(token)

        try:
            response = ' '.join(response)
        except Exception as e:
            # pprint(response)
            raise
        response = response.replace(' -s', 's')
        response = response.replace(' -ly', 'ly')
        response = response.replace(' .', '.')
        response = response.replace(' ?', '?')
        return response

    def model_predict(self, data_feed):
        ctx_lens = data_feed['context_lens']  # (batch_size, )
        short_ctx_utts = self.np2var(self.extract_short_ctx(
            data_feed['contexts'], ctx_lens), LONG)
        # (batch_size, max_ctx_len, max_utt_len)
        bs_label = self.np2var(data_feed['bs'], FLOAT)
        # (batch_size, max_ctx_len, max_utt_len)
        db_label = self.np2var(data_feed['db'], FLOAT)
        batch_size = len(ctx_lens)

        utt_summary, _, enc_outs = self.model.utt_encoder(
            short_ctx_utts.unsqueeze(1))

        enc_last = torch.cat(
            [bs_label, db_label, utt_summary.squeeze(1)], dim=1)

        mode = GEN

        logits_qy, log_qy = self.model.c2z(enc_last)
        sample_y = self.model.gumbel_connector(logits_qy, hard=mode == GEN)
        log_py = self.model.log_uniform_y

        # pack attention context
        if self.model.config.dec_use_attn:
            z_embeddings = torch.t(self.model.z_embedding.weight).split(
                self.model.k_size, dim=0)
            attn_context = []
            temp_sample_y = sample_y.view(-1, self.model.config.y_size,
                                          self.model.config.k_size)
            for z_id in range(self.model.y_size):
                attn_context.append(
                    torch.mm(temp_sample_y[:, z_id], z_embeddings[z_id]).unsqueeze(1))
            attn_context = torch.cat(attn_context, dim=1)
            dec_init_state = torch.sum(attn_context, dim=1).unsqueeze(0)
        else:
            dec_init_state = self.model.z_embedding(sample_y.view(
                1, -1, self.model.config.y_size * self.model.config.k_size))
            attn_context = None

        # decode
        if self.model.config.dec_rnn_cell == 'lstm':
            dec_init_state = tuple([dec_init_state, dec_init_state])

        dec_outputs, dec_hidden_state, ret_dict = self.model.decoder(batch_size=batch_size,
                                                                     dec_inputs=None,
                                                                     # (batch_size, response_size-1)
                                                                     # tuple: (h, c)
                                                                     dec_init_state=dec_init_state,
                                                                     attn_context=attn_context,
                                                                     # (batch_size, max_ctx_len, ctx_cell_size)
                                                                     mode=mode,
                                                                     gen_type='greedy',
                                                                     beam_size=self.model.config.beam_size)  # (batch_size, goal_nhid)

        # ret_dict['sample_z'] = sample_y
        # ret_dict['log_qy'] = log_qy

        pred_labels = [t.cpu().data.numpy()
                       for t in ret_dict[DecoderRNN.KEY_SEQUENCE]]
        pred_labels = np.array(
            pred_labels, dtype=int).squeeze(-1).swapaxes(0, 1)
        de_tknize = get_detokenize()
        for b_id in range(pred_labels.shape[0]):
            # only one val for pred_str now
            pred_str = get_sent(self.model.vocab, de_tknize, pred_labels, b_id)

            return pred_str


def get_sent(vocab, de_tknize, data, b_id, stop_eos=True, stop_pad=True):
    ws = []
    for t_id in range(data.shape[1]):
        w = vocab[data[b_id, t_id]]
        # TODO EOT
        if (stop_eos and w == EOS) or (stop_pad and w == PAD):
            break
        if w != PAD:
            ws.append(w)

    return de_tknize(ws)


def pad_to(max_len, tokens, do_pad):
    if len(tokens) >= max_len:
        return tokens[: max_len-1] + [tokens[-1]]
    elif do_pad:
        return tokens + [0] * (max_len - len(tokens))
    else:
        return tokens


def prepare_batch_gen(rows, config):
    domains = ['hotel', 'restaurant', 'train',
               'attraction', 'hospital', 'police', 'taxi']

    ctx_utts, ctx_lens = [], []
    out_utts, out_lens = [], []

    out_bs, out_db = [], []
    goals, goal_lens = [], [[] for _ in range(len(domains))]
    keys = []

    for row in rows:
        in_row, out_row = row['context'], row['response']

        # source context
        batch_ctx = []
        for turn in in_row:
            batch_ctx.append(
                pad_to(config.max_utt_len, turn['utt'], do_pad=True))
        ctx_utts.append(batch_ctx)
        ctx_lens.append(len(batch_ctx))

        out_bs.append(out_row['bs'])
        out_db.append(out_row['db'])

    batch_size = len(ctx_lens)
    vec_ctx_lens = np.array(ctx_lens)  # (batch_size, ), number of turns
    max_ctx_len = np.max(vec_ctx_lens)
    vec_ctx_utts = np.zeros(
        (batch_size, max_ctx_len, config.max_utt_len), dtype=np.int32)
    vec_out_bs = np.array(out_bs)  # (batch_size, 94)
    vec_out_db = np.array(out_db)  # (batch_size, 30)

    for b_id in range(batch_size):
        vec_ctx_utts[b_id, :vec_ctx_lens[b_id], :] = ctx_utts[b_id]

    return Pack(context_lens=vec_ctx_lens,  # (batch_size, )
                # (batch_size, max_ctx_len, max_utt_len)
                contexts=vec_ctx_utts,
                bs=vec_out_bs,  # (batch_size, 94)
                db=vec_out_db  # (batch_size, 30)
                )


if __name__ == '__main__':

    domain_name = 'object_division'
    domain_info = domain.get_domain(domain_name)
    
    train_data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/norm-multi-woz/train_dials.json')

    config = Pack(
        seed=10,
        train_path=train_data_path,
        max_vocab_size=1000,
        last_n_model=5,
        max_utt_len=50,
        max_dec_len=50,
        backward_size=2,
        batch_size=1,
        use_gpu=True,
        op='adam',
        init_lr=0.001,
        l2_norm=1e-05,
        momentum=0.0,
        grad_clip=5.0,
        dropout=0.5,
        max_epoch=100,
        embed_size=100,
        num_layers=1,
        utt_rnn_cell='gru',
        utt_cell_size=300,
        bi_utt_cell=True,
        enc_use_attn=True,
        dec_use_attn=True,
        dec_rnn_cell='lstm',
        dec_cell_size=300,
        dec_attn_mode='cat',
        y_size=10,
        k_size=20,
        beta=0.001,
        simple_posterior=True,
        contextual_posterior=True,
        use_mi=False,
        use_pr=True,
        use_diversity=False,
        #
        beam_size=20,
        fix_batch=True,
        fix_train_batch=False,
        avg_type='word',
        print_step=300,
        ckpt_step=1416,
        improve_threshold=0.996,
        patient_increase=2.0,
        save_model=True,
        early_stop=False,
        gen_type='greedy',
        preview_batch_num=None,
        k=domain_info.input_length(),
        init_range=0.1,
        pretrain_folder='2019-09-20-21-43-06-sl_cat',
        forward_only=False
    )

    state = {'belief_state': {
        "police": {
            "book": {
                "booked": []
            },
            "semi": {}
        },
        "hotel": {
            "book": {
                "booked": [],
                "people": "",
                "day": "",
                "stay": ""
            },
            "semi": {
                "name": "",
                "area": "",
                "parking": "",
                "pricerange": "",
                "stars": "",
                "internet": "",
                "type": ""
            }
        },
        "attraction": {
            "book": {
                "booked": []
            },
            "semi": {
                "type": "",
                "name": "",
                "area": ""
            }
        },
        "restaurant": {
            "book": {
                "booked": [],
                "people": "",
                "day": "",
                "time": ""
            },
            "semi": {
                "food": "",
                "pricerange": "",
                "name": "",
                "area": "west",
            }
        },
        "hospital": {
            "book": {
                "booked": []
            },
            "semi": {
                "department": ""
            }
        },
        "taxi": {
            "book": {
                "booked": []
            },
            "semi": {
                "leaveAt": "",
                "destination": "",
                "departure": "",
                "arriveBy": ""
            }
        },
        "train": {
            "book": {
                "booked": [],
                "people": ""
            },
            "semi": {
                "leaveAt": "",
                "destination": "",
                "day": "",
                "arriveBy": "",
                "departure": ""
            }
        }
    },
             'history': [['null', 'I want to find a restaurant west of town .']],
             'request_state': {},
             'user_action': {'Restaurant-Inform': [['area', 'west']]}}

    cur_model = LaRL(model_file="https://convlab.blob.core.windows.net/models/larl_model.zip")

    response = cur_model.predict(state)
    import pprint as pp
    pp.pprint(response)
