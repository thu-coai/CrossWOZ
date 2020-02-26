import requests
import zipfile
import argparse
import json
import os
import shutil
import time
import re
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, os.pardir))

import numpy as np
import torch

from convlab2.policy.policy import Policy
from convlab2.policy.mdrg.multiwoz.utils import delexicalize, util, dbquery, dbPointer
from convlab2.policy.mdrg.multiwoz.utils.nlp import normalize
from convlab2.policy.mdrg.multiwoz.evaluator import evaluateModel
from convlab2.policy.mdrg.multiwoz.mdrg_model import Model
from convlab2.policy.mdrg.multiwoz.create_delex_data import delexicaliseReferenceNumber, get_dial

from convlab2.util.multiwoz.state import default_state


def load_config(args):
    config = util.unicode_to_utf8(
        json.load(open(os.path.join(os.path.dirname(__file__), args.model_path + '.json'), 'rb')))
    for key, value in args.__args.items():
        try:
            config[key] = value.value
        except:
            config[key] = value

    return config


def loadModelAndData(num, args):
    # Load dictionaries
    with open(os.path.join(os.path.curdir(__file__),'data/input_lang.index2word.json')) as f:
        input_lang_index2word = json.load(f)
    with open(os.path.join(os.path.curdir(__file__),'data/input_lang.word2index.json')) as f:
        input_lang_word2index = json.load(f)
    with open(os.path.join(os.path.curdir(__file__),'data/output_lang.index2word.json')) as f:
        output_lang_index2word = json.load(f)
    with open(os.path.join(os.path.curdir(__file__),'data/output_lang.word2index.json')) as f:
        output_lang_word2index = json.load(f)

    # Reload existing checkpoint
    model = Model(args, input_lang_index2word, output_lang_index2word, input_lang_word2index, output_lang_word2index)
    if args.load_param:
        model.loadModel(iter=num)

    # Load data
    if os.path.exists(args.decode_output):
        shutil.rmtree(args.decode_output)
        os.makedirs(args.decode_output)
    else:
        os.makedirs(args.decode_output)

    if os.path.exists(args.valid_output):
        shutil.rmtree(args.valid_output)
        os.makedirs(args.valid_output)
    else:
        os.makedirs(args.valid_output)

    # Load validation file list:
    with open(os.path.join(os.path.curdir(__file__), 'data/val_dials.json')) as outfile:
        val_dials = json.load(outfile)

    # Load test file list:
    with open(os.path.join(os.path.curdir(__file__), 'data/test_dials.json')) as outfile:
        test_dials = json.load(outfile)
    return model, val_dials, test_dials


def addBookingPointer(task, turn, pointer_vector):
    """Add information about availability of the booking option."""
    # Booking pointer
    rest_vec = np.array([1, 0])
    if True:
        # if turn['metadata']['restaurant'].has_key("book"):
        if "book" in turn['metadata']['restaurant']:
            # if turn['metadata']['restaurant']['book'].has_key("booked"):
            if "booked" in turn['metadata']['restaurant']['book']:
                if turn['metadata']['restaurant']['book']["booked"]:
                    if "reference" in turn['metadata']['restaurant']['book']["booked"][0]:
                        rest_vec = np.array([0, 1])

    hotel_vec = np.array([1, 0])
    if True:
        # if turn['metadata']['hotel'].has_key("book"):
        if "book" in turn['metadata']['hotel']:
            # if turn['metadata']['hotel']['book'].has_key("booked"):
            if "booked" in turn['metadata']['hotel']['book']:
                if turn['metadata']['hotel']['book']["booked"]:
                    if "reference" in turn['metadata']['hotel']['book']["booked"][0]:
                        hotel_vec = np.array([0, 1])

    train_vec = np.array([1, 0])
    if True:
        # if turn['metadata']['train'].has_key("book"):
        if "book" in turn['metadata']['train']:
            # if turn['metadata']['train']['book'].has_key("booked"):
            if "booked" in turn['metadata']['train']['book']:
                if turn['metadata']['train']['book']["booked"]:
                    if "reference" in turn['metadata']['train']['book']["booked"][0]:
                        train_vec = np.array([0, 1])

    pointer_vector = np.append(pointer_vector, rest_vec)
    pointer_vector = np.append(pointer_vector, hotel_vec)
    pointer_vector = np.append(pointer_vector, train_vec)

    return pointer_vector

def addDBPointer(state):
    '''Create database pointer for all related domains.
    domains = ['restaurant', 'hotel', 'attraction', 'train']
    pointer_vector = np.zeros(6 * len(domains))
    for domain in domains:
        num_entities = dbPointer.queryResult(domain, turn)
        pointer_vector = dbPointer.oneHotVector(num_entities, domain, pointer_vector)

    return pointer_vector
    '''
    domains = ['restaurant', 'hotel', 'attraction', 'train']
    pointer_vector = np.zeros(6 * len(domains))
    db_results = {}
    num_entities = {}
    for domain in domains:
        # entities = dbPointer.queryResultVenues(domain, {'metadata': state})
        try:
            entities = dbquery.query(domain, state['metadata'][domain]['semi'].items())
        except:
            entities = dbquery.query(domain, state['belief_state'][domain]['semi'].items())
        num_entities[domain] = len(entities)
        if len(entities) > 0:
            # fields = dbPointer.table_schema(domain)
            # db_results[domain] = dict(zip(fields, entities[0]))
            db_results[domain] = entities[0]
        # pointer_vector = dbPointer.oneHotVector(len(entities), domain, pointer_vector)
        pointer_vector = dbPointer.oneHotVector(len(entities), domain, pointer_vector)

    return pointer_vector, db_results, num_entities


def decodeWrapper(args):
    # Load config file
    with open(os.path.join(os.path.curdir(__file__), args.model_path + '.config')) as f:
        add_args = json.load(f)
        for k, v in add_args.items():
            setattr(args, k, v)

        args.mode = 'test'
        args.load_param = True
        args.dropout = 0.0
        assert args.dropout == 0.0

    # Start going through models
    args.original = args.model_path
    for ii in range(1, args.no_models + 1):
        print(70 * '-' + 'EVALUATING EPOCH %s' % ii)
        args.model_path = args.model_path + '-' + str(ii)
        try:
            decode(ii)
        except:
            print('cannot decode')

        args.model_path = args.original


def get_active_domain(prev_active_domain, prev_state, state):
    domains = ['hotel', 'restaurant', 'attraction', 'train', 'taxi', 'hospital', 'police']
    active_domain = None
    # print('get_active_domain')
    for domain in domains:
        if domain not in prev_state and domain not in state:
            continue
        if domain in prev_state and domain not in state:
            return domain
        elif domain not in prev_state and domain in state:
            return domain
        elif prev_state[domain] != state[domain]:
            active_domain = domain
    if active_domain is None:
        active_domain = prev_active_domain
    return active_domain


def createDelexData(dialogue):
    """Main function of the script - loads delexical dictionary,
    goes through each dialogue and does:
    1) data normalization
    2) delexicalization
    3) addition of database pointer
    4) saves the delexicalized data
    """

    # create dictionary of delexicalied values that then we will search against, order matters here!
    dic = delexicalize.prepareSlotValuesIndependent()
    delex_data = {}

    # fin1 = open('data/multi-woz/data.json', 'r')
    # data = json.load(fin1)

        # dialogue = data[dialogue_name]
    dial = dialogue['cur']
    idx_acts = 1

    for idx, turn in enumerate(dial['log']):
        # print(idx)
        # print(turn)
        # normalization, split and delexicalization of the sentence
        sent = normalize(turn['text'])

        words = sent.split()
        sent = delexicalize.delexicalise(' '.join(words), dic)

        # parsing reference number GIVEN belief state
        sent = delexicaliseReferenceNumber(sent, turn)

        # changes to numbers only here
        digitpat = re.compile('\d+')
        sent = re.sub(digitpat, '[value_count]', sent)
        # print(sent)

        # delexicalized sentence added to the dialogue
        dial['log'][idx]['text'] = sent

        if idx % 2 == 1:  # if it's a system turn
            # add database pointer
            pointer_vector, db_results, num_entities = addDBPointer(turn)
            # add booking pointer
            pointer_vector = addBookingPointer(dial, turn, pointer_vector)

            # print pointer_vector
            dial['log'][idx - 1]['db_pointer'] = pointer_vector.tolist()

        idx_acts += 1
    dial = get_dial(dial)

    if dial:
        dialogue = {}
        dialogue['usr'] = []
        dialogue['sys'] = []
        dialogue['db'] = []
        dialogue['bs'] = []
        for turn in dial:
            # print(turn)
            dialogue['usr'].append(turn[0])
            dialogue['sys'].append(turn[1])
            dialogue['db'].append(turn[2])
            dialogue['bs'].append(turn[3])

    delex_data['cur'] = dialogue

    return delex_data


def populate_template(template, top_results, num_results, state):
    active_domain = None if len(top_results.keys()) == 0 else list(top_results.keys())[0]
    template = template.replace('book [value_count] of them', 'book one of them')
    tokens = template.split()
    response = []
    for index, token in enumerate(tokens):
        if token.startswith('[') and token.endswith(']'):
            domain = token[1:-1].split('_')[0]
            slot = token[1:-1].split('_')[1]
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
                    if 'arrive' in response:
                        for d in state:
                            if d == 'history':
                                continue
                            if 'destination' in state[d]['semi']:
                                response.append(state[d]['semi']['destination'])
                                break
                    elif 'leave' in response:
                        for d in state:
                            if d == 'history':
                                continue
                            if 'departure' in state[d]['semi']:
                                response.append(state[d]['semi']['departure'])
                                break
                    else:
                        try:
                            for d in state:
                                if d == 'history':
                                    continue
                                for s in ['destination', 'departure']:
                                    if s in state[d]['semi']:
                                        response.append(state[d]['semi'][s])
                                        raise Exception
                        except Exception:
                            pass
                        else:
                            response.append(token)
                elif slot == 'time':
                    if 'arrive' in ' '.join(response[-3:]):
                        if active_domain is not None and 'arriveBy' in top_results[active_domain]:
                            # print('{} -> {}'.format(token, top_results[active_domain]['arriveBy']))
                            response.append(top_results[active_domain]['arriveBy'])
                            continue
                        for d in state:
                            if d == 'history':
                                continue
                            if 'arriveBy' in state[d]['semi']:
                                response.append(state[d]['semi']['arriveBy'])
                                break
                    elif 'leave' in ' '.join(response[-3:]):
                        if active_domain is not None and 'leaveAt' in top_results[active_domain]:
                            # print('{} -> {}'.format(token, top_results[active_domain]['leaveAt']))
                            response.append(top_results[active_domain]['leaveAt'])
                            continue
                        for d in state:
                            if d == 'history':
                                continue
                            if 'leaveAt' in state[d]['semi']:
                                response.append(state[d]['semi']['leaveAt'])
                                break
                    elif 'book' in response:
                        if state['restaurant']['book']['time'] != "":
                            response.append(state['restaurant']['book']['time'])
                    else:
                        try:
                            for d in state:
                                if d == 'history':
                                    continue
                                for s in ['arriveBy', 'leaveAt']:
                                    if s in state[d]['semi']:
                                        response.append(state[d]['semi'][s])
                                        raise Exception
                        except Exception:
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
                elif domain == 'police':
                    if slot == 'phone':
                        response.append('01223358966')
                    elif slot == 'name':
                        response.append('Parkside Police Station')
                    elif slot == 'address':
                        response.append('Parkside, Cambridge')
                elif domain == 'taxi':
                    if slot == 'phone':
                        response.append('01223358966')
                    elif slot == 'color':
                        response.append('white')
                    elif slot == 'type':
                        response.append('toyota')
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

def decode(data, model, device):
    # model, val_dials, test_dials = loadModelAndData(num)
    # device = torch.device("cuda" if args.cuda else "cpu")

    for ii in range(1):
        if ii == 0:
            # print(50 * '-' + 'GREEDY')
            model.beam_search = False
        else:
            print(50 * '-' + 'BEAM')
            model.beam_search = True

        # VALIDATION
        val_dials_gen = {}
        valid_loss = 0
        # for name, val_file in val_dials.items():
        for i in range(1):
            val_file = data['cur']
            input_tensor = [];  target_tensor = [];bs_tensor = [];db_tensor = []
            input_tensor, target_tensor, bs_tensor, db_tensor = util.loadDialogue(model, val_file, input_tensor, target_tensor, bs_tensor, db_tensor)
            # create an empty matrix with padding tokens
            input_tensor, input_lengths = util.padSequence(input_tensor)
            target_tensor, target_lengths = util.padSequence(target_tensor)
            bs_tensor = torch.tensor(bs_tensor, dtype=torch.float, device=device)
            db_tensor = torch.tensor(db_tensor, dtype=torch.float, device=device)

            output_words, loss_sentence = model.predict(input_tensor, input_lengths, target_tensor, target_lengths,
                                                        db_tensor, bs_tensor)

            valid_loss += 0
            return output_words[-1]


def loadModel(num, args):

    # Load dictionaries
    with open(os.path.join(os.path.dirname(__file__), 'data','input_lang.index2word.json')) as f:
        input_lang_index2word = json.load(f)
    with open(os.path.join(os.path.dirname(__file__), 'data', 'input_lang.word2index.json')) as f:
        input_lang_word2index = json.load(f)
    with open(os.path.join(os.path.dirname(__file__), 'data', 'output_lang.index2word.json')) as f:
        output_lang_index2word = json.load(f)
    with open(os.path.join(os.path.dirname(__file__), 'data', 'output_lang.word2index.json')) as f:
        output_lang_word2index = json.load(f)

    # Reload existing checkpoint
    model = Model(args, input_lang_index2word, output_lang_index2word, input_lang_word2index, output_lang_word2index)

    if args.load_param:
        model.loadModel(iter=num)
    return model


class MDRGWordPolicy(Policy):
    def __init__(self, num=1):
        parser = argparse.ArgumentParser(description='S2S')
        parser.add_argument('--no_cuda', type=util.str2bool, nargs='?', const=True, default=True,
                            help='enables CUDA training')
        parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

        parser.add_argument('--no_models', type=int, default=20, help='how many models to evaluate')
        parser.add_argument('--original', type=str, default='model/model/', help='Original path.')

        parser.add_argument('--dropout', type=float, default=0.0)
        parser.add_argument('--use_emb', type=str, default='False')

        parser.add_argument('--beam_width', type=int, default=10, help='Beam width used in beamsearch')
        parser.add_argument('--write_n_best', type=util.str2bool, nargs='?', const=True, default=False,
                            help='Write n-best list (n=beam_width)')

        parser.add_argument('--model_path', type=str, default='model/model/translate.ckpt',
                            help='Path to a specific model checkpoint.')
        parser.add_argument('--model_dir', type=str, default='data/multi-woz/model/model/')
        parser.add_argument('--model_name', type=str, default='translate.ckpt')

        parser.add_argument('--valid_output', type=str, default='model/data/val_dials/',
                            help='Validation Decoding output dir path')
        parser.add_argument('--decode_output', type=str, default='model/data/test_dials/',
                            help='Decoding output dir path')


        args = parser.parse_args([])

        args.cuda = not args.no_cuda and torch.cuda.is_available()

        torch.manual_seed(args.seed)

        self.device = torch.device("cuda" if args.cuda else "cpu")
        with open(os.path.join(os.path.dirname(__file__), args.model_path + '.config'), 'r') as f:
            add_args = json.load(f)
            # print(add_args)
            for k, v in add_args.items():
                setattr(args, k, v)
            # print(args)
            args.mode = 'test'
            args.load_param = True
            args.dropout = 0.0
            assert args.dropout == 0.0

        # Start going through models
        args.original = args.model_path
        args.model_path = args.original
        self.model = loadModel(num, args)
        self.dial = {"cur": {"log": []}}
        self.prev_state = default_state()
        self.prev_active_domain = None


    def predict(self, state):
        # active_domain = get_active_domain(self.prev_active_domain, self.prev_state['belief_state'], state['belief_state'])
        # print(state)
        # print(active_domain)

        pointer_vector, top_results, num_results = addDBPointer(state)
        # print(num_results)
        last_usr_uttr = state['history'][-1][-1]
        usr_turn = {"text": last_usr_uttr, "metadata": ""}
        sys_turn = {"text": "placeholder " * 50, "metadata": state['belief_state']}
        self.dial['cur']['log'].append(usr_turn)
        self.dial['cur']['log'].append(sys_turn)
        # print(self.dial)

        self.normalized_dial = createDelexData(self.dial)

        response = decode(self.normalized_dial, self.model, self.device)

        active_domain = None
        domains = ['restaurant', 'hotel', 'taxi', 'train', 'police', 'hospital', 'attraction']

        for word in response.split(' '):
            for domain in domains:
                if (domain + '_') in word:
                    active_domain = domain
                    break
            if active_domain is not None:
                break

        if active_domain is not None and active_domain in num_results:
            num_results = num_results[active_domain]
        else:
            num_results = 0
        if active_domain is not None and active_domain in top_results:
            top_results = {active_domain: top_results[active_domain]}
        else:
            top_results = {}
        # response = populate_template(output_words[0], top_results, num_results, state)
        # return response, active_domain

        # print(response)
        response = populate_template(response, top_results, num_results, state['belief_state'])
        response = response.split(' ')
        if '_UNK' in response:
            response.remove('_UNK')
        response = ' '.join(response)
        if not top_results:
            response = 'Sorry, I can\'t find any result matching your condition, please try again.'

        self.dial['cur']['log'][-1]['text'] = response

        return response

    def init_session(self):
        self.dial = {"cur": {"log": []}}

def main():
    s = default_state()
    s['history'] = [['null', 'I want a chinese restaurant']]
    # s['belief_state']['attraction']['semi']['area'] = 'centre'
    s['belief_state']['restaurant']['semi']['area'] = 'south'
    s['belief_state']['restaurant']['semi']['food'] = 'mexican'
    testPolicy = MDRGWordPolicy()
    print(s)
    print(testPolicy.predict(s))


if __name__ == '__main__':
    main()
