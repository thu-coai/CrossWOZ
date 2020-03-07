import json
import torch
import torch.utils.data as data
import unicodedata
import string
import re
import random
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from utils.config import *
import logging 
import datetime

MEM_TOKEN_SIZE = 4

class Lang:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {1: "PAD", 3: "SOS", 2: "EOS", 0: 'UNK'}
        self.n_words = 4 # Count default tokens
    
    def index_words(self, story, trg=False):
        if trg:
            for word in story.split(' '):
                self.index_word(word)
        else:
            for word_triple in story:
                for word in word_triple:
                    self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    # data_item = {'content_arr':content_arr, 'bot_action':bot_action, 'bot_action_idx':bot_action_idx,
    #                 'ent_query':ent_query, 'ent_query_idx':ent_query_idx, 'gold_response':gold_response}
    def __init__(self, data_item, src_word2id, trg_word2id, max_len, query2idx):
        """Reads source and target sequences from txt files."""
        self.dialID = data_item['dialID']
        self.turnID = data_item['turnID']
        self.content_arr = data_item['content_arr']
        self.bot_action_idx = data_item['bot_action_idx']
        self.bot_action = data_item['bot_action']
        self.ent_query = data_item['ent_query']
        self.ent_query_idx = data_item['ent_query_idx']
        self.gold_response = data_item['gold_response']     
        self.src_plain = data_item['content_arr']
        self.num_total_seqs = len(self.content_arr)
        self.src_word2id = src_word2id
        self.trg_word2id = trg_word2id
        self.max_len = max_len
        self.query2idx = query2idx

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        dialID = self.dialID[index]
        turnID = self.turnID[index]
        content_arr = self.content_arr[index]
        bot_action_idx = self.bot_action_idx[index]
        bot_action = self.bot_action[index]
        ent_query = self.ent_query[index]
        ent_query_idx = self.ent_query_idx[index]
        gold_response = self.gold_response[index]
        src_plain  = self.content_arr[index]

        content_arr = self.preprocess(content_arr, self.src_word2id)
        q, q_idx = self.preprocess_query(ent_query_idx, ent_query)

        item_dict = {'dialID':dialID,'turnID':turnID,'content_arr':content_arr, 'bot_action_idx':bot_action_idx, 'bot_action':bot_action, 
            'ent_query':q, 'ent_query_idx':q_idx, 'gold_response':gold_response, 'src_plain':src_plain}
        return item_dict

    def __len__(self):
        return self.num_total_seqs

    def preprocess(self, sequence, word2idx):
        """Converts words to idx."""
        story = []
        for i, word_triple in enumerate(sequence):
            story.append([])
            for ii, word in enumerate(word_triple):
                temp = word2idx[word] if word in word2idx else UNK_token
                story[i].append(temp)
        try:
            story = torch.LongTensor(story)
        except:
            print("Cannot change to tensor...")
            exit(1)
        return story
    
    def preprocess_query(self, ent_query_idx, ent_query):
        """Converts entity query to idx."""
        ent = ent_query[0]
        q = [self.query2idx[ent]]
        q_idx = ent_query_idx[1]
        return q, q_idx

def collate_fn(data):
    
    def merge(sequences,max_len):
        lengths = [len(seq) for seq in sequences]
        if (max_len):
            padded_seqs = torch.ones(len(sequences), max(lengths), MEM_TOKEN_SIZE).long()
            for i, seq in enumerate(sequences):
                end = lengths[i]
                if len(seq) != 0:
                    padded_seqs[i,:end,:] = seq[:end]
        else:
            padded_seqs = torch.ones(len(sequences), max(lengths)).long()
            for i, seq in enumerate(sequences):
                end = lengths[i]
                padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x['content_arr']), reverse=True)
    # seperate source and target sequences
    #src_seqs, trg_seqs, ind_seqs, target_plain, max_len, src_plain = zip(*data)
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    # merge sequences (from tuple of 1D tensor to 2D tensor)
    content_arr, content_arr_len = merge(item_info['content_arr'],True)
    #ind_seqs, ind_lenght = merge(ind_seqs, None)
    
    content_arr = Variable(content_arr).transpose(0,1)
    bot_action_idx = Variable(torch.LongTensor(item_info['bot_action_idx']))
    ent_query = Variable(torch.LongTensor(item_info['ent_query']))
    ent_query_idx = Variable(torch.LongTensor(item_info['ent_query_idx'])) 
    
    if USE_CUDA:
        content_arr = content_arr.cuda()
        bot_action_idx = bot_action_idx.cuda()
        ent_query = ent_query.cuda()
        ent_query_idx = ent_query_idx.cuda()

    data_form = {'dailID':item_info['dialID'],'turnID':item_info['turnID'],'content_arr':content_arr, 'content_arr_len':content_arr_len, 
            'bot_action_idx':bot_action_idx, 'bot_action':item_info['bot_action'], 
            'ent_query':ent_query, 'ent_query_idx':ent_query_idx, 'gold_response':item_info['gold_response'], 
            'src_plain':item_info['src_plain']}

    return data_form

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn' )

# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
    
    s = unicode_to_ascii(s.lower().strip())
    if s=='<silence>':
        return s
    s = re.sub(r"([,.!?])", r" \1 ", s)
    s = re.sub(r"[^a-zA-Z,.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

def read_langs(file_name, entity, cand2DLidx, idx2candDL, max_line = None):
    logging.info(("Reading lines from {}".format(file_name)))
    data=[]
    content_arr = []
    #conversation_arr = []
    u=None
    r=None
    user_counter = 0
    system_counter = 0
    system_res_counter = 0
    KB_counter = 0
    dialog_counter = 0
    with open(file_name) as fin:
        #cnt_ptr = 0
        #cnt_voc = 0
        max_r_len = 0
        cnt_lin = 1
        time_counter = 1 
        for line in fin:
            line=line.strip()
            if line:
                nid, line = line.split(' ', 1)
                if '\t' in line:
                    u, r = line.split('\t')
                    if u!='<SILENCE>': user_counter += 1
                    system_counter += 1
                    bot_action_idx = cand2DLidx[r]
                    bot_action = idx2candDL[bot_action_idx]

                    gen_u = generate_memory(u, "$u", str(time_counter)) 
                    content_arr += gen_u
                    #conversation_arr += gen_u
                    
                    ent_query = {}
                    ent_query_idx = {}
                    for idx, key in enumerate(r.split(' ')):
                        if (key in entity):
                            index = [loc for loc, val in enumerate(content_arr) if (val[0] == key)]
                            if (index):
                                index = max(index)
                                #cnt_ptr += 1
                                ent_query_idx[bot_action.split(' ')[idx]] = index
                                ent_query[bot_action.split(' ')[idx]] = key
                            else:
                                print('[Wrong] Cannot find the entity')
                                exit(1)
                        system_res_counter += 1 
                    
                    if ent_query == {}:
                        ent_query = {'UNK':'$$$$'}
                        ent_query_idx = {'UNK': len(content_arr)}
                        content_arr_temp = content_arr + [['$$$$']*MEM_TOKEN_SIZE]
                    else:
                        content_arr_temp = content_arr
                    # ent = []
                    # for key in r.split(' '):
                    #     if(key in entity):
                    #         ent.append(key)

                    for ent in ent_query.keys():
                        data_item = {'dialID':dialog_counter,'turnID':system_counter,'content_arr':content_arr_temp, 'bot_action':bot_action, 'bot_action_idx':bot_action_idx,
                            'ent_query':[ent,ent_query[ent]], 'ent_query_idx':[ent,ent_query_idx[ent]], 'gold_response':r}
                        data.append(data_item)


                    #data.append([content_arr_temp,r,r_index,conversation_arr,ent])
                    gen_r = generate_memory(r, "$s", str(time_counter)) 
                    content_arr += gen_r
                    #conversation_arr += gen_r
                    time_counter += 1
                else:
                    KB_counter += 1
                    r=line
                    content_arr += generate_memory(r, "", "")  
            else:
                cnt_lin+=1
                if(max_line and cnt_lin>=max_line):
                    break
                content_arr=[]
                content_arr_temp = []
                #conversation_arr = []
                time_counter = 1
                dialog_counter += 1
    max_len = max([len(d['content_arr']) for d in data])
    logging.info("Nb of dialogs = {} ".format(dialog_counter))
    #logging.info("Pointer percentace= {} ".format(cnt_ptr/(cnt_ptr+cnt_voc)))
    logging.info("Max responce Len: {}".format(max_r_len))
    logging.info("Max Input Len: {}".format(max_len))
    logging.info("Avg. User Utterances: {}".format(user_counter*1.0/dialog_counter))
    logging.info("Avg. Bot Utterances: {}".format(system_counter*1.0/dialog_counter))
    logging.info("Avg. KB results: {}".format(KB_counter*1.0/dialog_counter))
    logging.info("Avg. responce Len: {}".format(system_res_counter*1.0/system_counter))
    
    print('Sample: ',data[5])
    return data, max_len

def generate_memory(sent, speaker, time):
    sent_new = []
    sent_token = sent.split(' ')
    if speaker=="$u" or speaker=="$s":
        for idx, word in enumerate(sent_token):
            temp = [word, speaker, 'turn'+str(time), 'word'+str(idx)] + ["PAD"]*(MEM_TOKEN_SIZE-4)
            sent_new.append(temp)
    else:
        if sent_token[1]=="R_rating":
            sent_token = sent_token + ["PAD"]*(MEM_TOKEN_SIZE-len(sent_token))
        else:
            sent_token = sent_token[::-1] + ["PAD"]*(MEM_TOKEN_SIZE-len(sent_token))
        sent_new.append(sent_token)
    return sent_new

def get_seq(pairs,lang,batch_size,type,max_len, query2idx):   
    dialID_arr = []
    turnID_arr = []
    content_arr = []
    bot_action = []
    bot_action_idx = []
    ent_query = []
    ent_query_idx = []
    gold_response = []
    
    for pair in pairs:
        dialID_arr.append(pair['dialID'])
        turnID_arr.append(pair['turnID'])
        content_arr.append(pair['content_arr'])
        bot_action.append(pair['bot_action'])
        bot_action_idx.append(pair['bot_action_idx'])
        ent_query.append(pair['ent_query'])
        ent_query_idx.append(pair['ent_query_idx'])
        gold_response.append(pair['gold_response'])
        if(type):
            lang.index_words(pair['content_arr'])
            lang.index_words(pair['bot_action'], trg=True)
    
    data_item = {'dialID':dialID_arr, 'turnID':turnID_arr, 'content_arr':content_arr, 'bot_action':bot_action, 'bot_action_idx':bot_action_idx,
                'ent_query':ent_query, 'ent_query_idx':ent_query_idx, 'gold_response':gold_response}

    dataset = Dataset(data_item, lang.word2index, lang.word2index, max_len, query2idx)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=type,
                                              collate_fn=collate_fn)
    return data_loader

def get_type_dict(kb_path, dstc2=False): 
    """
    Specifically, we augment the vocabulary with some special words, one for each of the KB entity types 
    For each type, the corresponding type word is added to the candidate representation if a word is found that appears 
    1) as a KB entity of that type, 
    """
    type_dict = {'R_restaurant':[]}

    kb_path_temp = kb_path
    fd = open(kb_path_temp,'r') 

    for line in fd:
        if dstc2:
            x = line.replace('\n','').split(' ')
            rest_name = x[1]
            entity = x[2]
            entity_value = x[3]
        else:
            x = line.split('\t')[0].split(' ')
            rest_name = x[1]
            entity = x[2]
            entity_value = line.split('\t')[1].replace('\n','')
    
        if rest_name not in type_dict['R_restaurant']:
            type_dict['R_restaurant'].append(rest_name)
        if entity not in type_dict.keys():
            type_dict[entity] = []
        if entity_value not in type_dict[entity]:
            type_dict[entity].append(entity_value)
    return type_dict

def entityList(kb_path, task_id):
    type_dict = get_type_dict(kb_path, dstc2=(task_id==6))
    entity_list = []
    for key in type_dict.keys():
        for value in type_dict[key]:
            entity_list.append(value)
    return entity_list


def load_candidates(task_id, candidates_f):
    # containers
    #type_dict = get_type_dict(KB_DIR, dstc2=(task_id==6))
    candidates, candid2idx, idx2candid = [], {}, {}
    # update data source file based on task id
    #candidates_f = DATA_SOURCE_TASK6 if task_id==6 else candidates_f

    # read from file
    with open(candidates_f) as f:
        # iterate through lines
        for i, line in enumerate(f):
            # tokenize each line into... well.. tokens!
            temp = line.strip().split(' ')
            candid2idx[line.strip().split(' ',1)[1]] = i
            candidates.append(temp[1:])
            idx2candid[i] = line.strip().split(' ',1)[1]
    return candidates, candid2idx, idx2candid

def candid2DL(candid_path, kb_path, task_id):
    type_dict = get_type_dict(kb_path, dstc2=(task_id==6))
    ent_list = entityList(kb_path,int(task_id))
    candidates, _, _ = load_candidates(task_id=task_id, candidates_f=candid_path)
    candid_all = []  
    candid2candDL = {}
    for index, cand in enumerate(candidates):
        cand_DL = [ x for x in cand]
        for index, word in enumerate(cand_DL):
            if word in ent_list:
                for type_name in type_dict:
                    if word in type_dict[type_name] and type_name != 'R_rating':
                        cand_DL[index] = type_name
                        break
        cand_DL = ' '.join(cand_DL)
        candid_all.append(cand_DL)
        candid2candDL[' '.join(cand)] = cand_DL
    cand_list = list(set(candid_all))
    candDL2idx = dict((c, i) for i, c in enumerate(cand_list))
    idx2candDL = dict((i, c) for c, i in candDL2idx.items()) 

    cand2DLidx = {}
    for key in candid2candDL.keys():
        cand2DLidx[key] = candDL2idx[candid2candDL[key]]
        
    return cand2DLidx, idx2candDL


def prepare_data_seq(task, batch_size=100):
    file_train = 'data/dialog-bAbI-tasks/dialog-babi-task{}trn.txt'.format(task)
    file_dev = 'data/dialog-bAbI-tasks/dialog-babi-task{}dev.txt'.format(task)
    file_test = 'data/dialog-bAbI-tasks/dialog-babi-task{}tst.txt'.format(task)
    if (int(task) != 6):
        file_test_OOV = 'data/dialog-bAbI-tasks/dialog-babi-task{}tst-OOV.txt'.format(task)
        candid_file_path = 'data/dialog-bAbI-tasks/dialog-babi-candidates.txt'
        kb_path = 'data/dialog-bAbI-tasks/dialog-babi-kb-all.txt'
    else:
        candid_file_path = 'data/dialog-bAbI-tasks/dialog-babi-task6-dstc2-candidates.txt'
        kb_path = 'data/dialog-bAbI-tasks/dialog-babi-task6-dstc2-kb.txt'
    
    query2idx = {'UNK':0, 'R_restaurant':7, 'R_cuisine':1, 'R_location':2, 'R_price':3, 'R_number':4, 
                'R_phone':5, 'R_address':6}

    ent = entityList(kb_path,int(task))
    cand2DLidx, idx2candDL = candid2DL(candid_file_path, kb_path, int(task))

    pair_train, max_len_train = read_langs(file_train, ent, cand2DLidx, idx2candDL, max_line=None)
    pair_dev,max_len_dev = read_langs(file_dev, ent, cand2DLidx, idx2candDL, max_line=None)
    pair_test,max_len_test = read_langs(file_test, ent, cand2DLidx, idx2candDL, max_line=None)

    max_r_test_OOV = 0
    if (int(task) != 6):
        pair_test_OOV,max_len_test_OOV = read_langs(file_test_OOV, ent, cand2DLidx, idx2candDL, max_line=None)

    max_len = max(max_len_train,max_len_dev,max_len_test,max_len_test_OOV) +1
    max_r  = -1 #max(max_r_train,max_r_dev,max_r_test,max_r_test_OOV) +1
    lang = Lang()
    
    train = get_seq(pair_train,lang,batch_size,True,max_len, query2idx)
    dev   = get_seq(pair_dev,lang,batch_size,False,max_len, query2idx)
    test  = get_seq(pair_test,lang,batch_size,False,max_len, query2idx)
    
    if (int(task) != 6):
        testOOV = get_seq(pair_test_OOV,lang,batch_size,False,max_len, query2idx)
    else:
        testOOV = []
    
    logging.info("Read %s sentence pairs train" % len(pair_train))
    logging.info("Read %s sentence pairs dev" % len(pair_dev))
    logging.info("Read %s sentence pairs test" % len(pair_test))
    if (int(task) != 6):
        logging.info("Read %s sentence pairs testoov" % len(pair_test_OOV))    
    logging.info("Max len Input %s " % max_len)
    logging.info("Vocab_size %s " % lang.n_words)
    logging.info("USE_CUDA={}".format(USE_CUDA))

    return train, dev, test, testOOV, lang, max_len, max_r, idx2candDL, query2idx

if __name__=="__main__":
    prepare_data_seq(task=1,batch_size=1)