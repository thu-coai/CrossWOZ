import re
import json
import torch
import os
import zipfile
import pickle
from convlab2.policy.hdsa.multiwoz.transformer.Transformer import TableSemanticDecoder
from convlab2.policy.hdsa.multiwoz.transformer import Constants
from convlab2.util.file_util import cached_path

timepat = re.compile("\d{1,2}[:]\d{1,2}")
pricepat = re.compile("\d{1,3}[.]\d{1,2}")


def insertSpace(token, text):
    sidx = 0
    while True:
        sidx = text.find(token, sidx)
        if sidx == -1:
            break
        if sidx + 1 < len(text) and re.match('[0-9]', text[sidx - 1]) and \
                re.match('[0-9]', text[sidx + 1]):
            sidx += 1
            continue
        if text[sidx - 1] != ' ':
            text = text[:sidx] + ' ' + text[sidx:]
            sidx += 1
        if sidx + len(token) < len(text) and text[sidx + len(token)] != ' ':
            text = text[:sidx + 1] + ' ' + text[sidx + 1:]
        sidx += 1
    return text

def normalize(text, replacements, sub=True):
    # lower case every word
    text = text.lower()

    # replace white spaces in front and end
    text = re.sub(r'^\s*|\s*$', '', text)

    # hotel domain pfb30
    text = re.sub(r"b&b", "bed and breakfast", text)
    text = re.sub(r"b and b", "bed and breakfast", text)

    # normalize phone number
    ms = re.findall('\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4,5})', text)
    if ms:
        sidx = 0
        for m in ms:
            sidx = text.find(m[0], sidx)
            if text[sidx - 1] == '(':
                sidx -= 1
            eidx = text.find(m[-1], sidx) + len(m[-1])
            text = text.replace(text[sidx:eidx], ''.join(m))

    # normalize postcode
    ms = re.findall('([a-z]{1}[\. ]?[a-z]{1}[\. ]?\d{1,2}[, ]+\d{1}[\. ]?[a-z]{1}[\. ]?[a-z]{1}|[a-z]{2}\d{2}[a-z]{2})',
                    text)
    if ms:
        sidx = 0
        for m in ms:
            sidx = text.find(m, sidx)
            eidx = sidx + len(m)
            text = text[:sidx] + re.sub('[,\. ]', '', m) + text[eidx:]

    # weird unicode bug
    text = re.sub(u"(\u2018|\u2019)", "'", text)

    # replace time and and price
    if sub:
        text = re.sub(timepat, ' [value_time] ', text)
        text = re.sub(pricepat, ' [train_price] ', text)
        #text = re.sub(pricepat2, '[value_price]', text)

    # replace st.
    text = text.replace(';', ',')
    text = re.sub('$\/', '', text)
    text = text.replace('/', ' and ')

    # replace other special characters
    text = text.replace('-', ' ')
    text = re.sub('[\":\<>@\(\)]', '', text)

    # insert white space before and after tokens:
    for token in ['?', '.', ',', '!']:
        text = insertSpace(token, text)

    # insert white space for 's
    text = insertSpace('\'s', text)

    # replace it's, does't, you'd ... etc
    text = re.sub('^\'', '', text)
    text = re.sub('\'$', '', text)
    text = re.sub('\'\s', ' ', text)
    text = re.sub('\s\'', ' ', text)
    for fromx, tox in replacements:
        text = ' ' + text + ' '
        text = text.replace(fromx, tox)[1:-1]

    # remove multiple spaces
    text = re.sub(' +', ' ', text)

    # concatenate numbers
    tokens = text.split()
    i = 1
    while i < len(tokens):
        if re.match(u'^\d+$', tokens[i]) and \
                re.match(u'\d+$', tokens[i - 1]):
            tokens[i - 1] += tokens[i]
            del tokens[i]
        else:
            i += 1
    text = ' '.join(tokens)

    return text

def delexicaliseReferenceNumber(sent, turn, replacements):
    """Based on the belief state, we can find reference number that
    during data gathering was created randomly."""
    for domain in turn:
        if turn[domain]['book']['booked']:
            for slot in turn[domain]['book']['booked'][0]:
                if slot == 'reference':
                    val = '[' + domain + '_' + slot + ']'
                else:
                    val = '[' + domain + '_' + slot + ']'
                key = normalize(turn[domain]['book']['booked'][0][slot], replacements)
                sent = (' ' + sent + ' ').replace(' ' + key + ' ', ' ' + val + ' ')

                # try reference with hashtag
                key = normalize("#" + turn[domain]['book']['booked'][0][slot], replacements)
                sent = (' ' + sent + ' ').replace(' ' + key + ' ', ' ' + val + ' ')

                # try reference with ref#
                key = normalize("ref#" + turn[domain]['book']['booked'][0][slot], replacements)
                sent = (' ' + sent + ' ').replace(' ' + key + ' ', ' ' + val + ' ')
    return sent

def delexicalise(utt, dictionary):
    for key, val in dictionary:
        utt = (' ' + utt + ' ').replace(' ' + key + ' ', ' ' + val + ' ')
        utt = utt[1:-1]  # why this?

    return utt

def denormalize(uttr):
    uttr = uttr.replace(' -s', 's')
    uttr = uttr.replace('expensive -ly', 'expensive')
    uttr = uttr.replace('cheap -ly', 'cheap')
    uttr = uttr.replace('moderate -ly', 'moderately')
    uttr = uttr.replace(' -er', 'er')
    uttr = uttr.replace('_UNK', 'unknown')
    return uttr

class Tokenizer(object):
    def __init__(self, vocab, ivocab, use_field, lower_case=True):
        super(Tokenizer, self).__init__()
        self.lower_case = lower_case
        self.ivocab = ivocab
        self.vocab = vocab
        self.use_field = use_field
        if use_field:
            with open('data/placeholder.json') as f:
                self.fields = json.load(f)['field']
        
        self.vocab_len = len(self.vocab)

    def tokenize(self, sent):
        if self.lower_case:
            return sent.lower().split()
        else:
            return sent.split()

    def get_word_id(self, w, template=None):
        if self.use_field and template:
            if w in self.fields and w in template:
                return template.index(w) + self.vocab_len
        
        if w in self.vocab:
            return self.vocab[w]
        else:
            return self.vocab[Constants.UNK_WORD]
        
    
    def get_word(self, k, template=None):
        if k > self.vocab_len and self.use_field and template:
            return template[k - self.vocab_len]
        else:
            k = str(k)
            return self.ivocab[k]
            
    def convert_tokens_to_ids(self, sent, template=None):
        return [self.get_word_id(w, template) for w in sent]

    def convert_id_to_tokens(self, word_ids, template_ids=None, remain_eos=False):
        if isinstance(word_ids, list):
            if remain_eos:
                return " ".join([self.get_word(wid, None) for wid in word_ids 
                                 if wid != Constants.PAD])
            else:
                return " ".join([self.get_word(wid, None) for wid in word_ids 
                                 if wid not in [Constants.PAD, Constants.EOS] ])                
        else:
            if remain_eos:
                return " ".join([self.get_word(wid.item(), None) for wid in word_ids 
                                 if wid != Constants.PAD])
            else:
                return " ".join([self.get_word(wid.item(), None) for wid in word_ids 
                                 if wid not in [Constants.PAD, Constants.EOS]])
            
    def convert_template(self, template_ids):
        return [self.get_word(wid) for wid in template_ids if wid != Constants.PAD]

class HDSA_generator():
    def __init__(self, archive_file, model_file=None, use_cuda=False):
        if not os.path.isfile(archive_file):
            if not model_file:
              raise Exception("No model for HDSA is specified!")
            archive_file = cached_path(model_file)
        model_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.exists(os.path.join(model_dir, 'checkpoints')):
              archive = zipfile.ZipFile(archive_file, 'r')
              archive.extractall(model_dir)
        
            
        with open(os.path.join(model_dir, "data/vocab.json"), 'r') as f:
            vocabulary = json.load(f)
        
        vocab, ivocab = vocabulary['vocab'], vocabulary['rev']
        self.tokenizer = Tokenizer(vocab, ivocab, False)
        self.max_seq_length = 50
        self.history = []
        self.kbs = {}
        self.decoder = TableSemanticDecoder(vocab_size=self.tokenizer.vocab_len, d_word_vec=128, n_layers=3, 
                              d_model=128, n_head=4, dropout=0.2)
        self.device = 'cuda' if use_cuda else 'cpu'
        self.decoder.to(self.device)
        checkpoint_file = os.path.join(model_dir, "checkpoints/generator/BERT_dim128_w_domain")
        self.decoder.load_state_dict(torch.load(checkpoint_file))
        
        with open(os.path.join(model_dir, 'data/svdic.pkl'), 'rb') as f:
            self.dic = pickle.load(f)
            
        with open(os.path.join(model_dir, 'data/mapping.pair')) as f:
            self.replacements = []
            for line in f:
                tok_from, tok_to = line.replace('\n', '').split('\t')
                self.replacements.append((' ' + tok_from + ' ', ' ' + tok_to + ' '))
    
    def init_session(self):
        self.history = []
        self.kbs = {}

    def generate(self, state, pred_hierachical_act_vecs, kb):
        self.kbs[kb['domain']] = kb
        
        usr_post = state['history'][-1][-1]
        usr = delexicalise(' '.join(usr_post.split()), self.dic)
    
        # parsing reference number GIVEN belief state
        usr = delexicaliseReferenceNumber(usr, state['belief_state'], self.replacements)
    
        # changes to numbers only here
        digitpat = re.compile('\d+')
        usr = re.sub(digitpat, '[value_count]', usr)
        
        tokens = self.tokenizer.tokenize(usr)
        if self.history:
            tokens = self.history + [Constants.SEP_WORD] + tokens
        if len(tokens) > self.max_seq_length - 2:
            tokens = tokens[-(self.max_seq_length - 2):]
        tokens = [Constants.CLS_WORD] + tokens + [Constants.SEP_WORD]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)

        hyps = self.decoder.translate_batch(act_vecs=pred_hierachical_act_vecs, src_seq=input_ids, 
                                       n_bm=2, max_token_seq_len=40)
        pred = self.tokenizer.convert_id_to_tokens(hyps[0])
        
        if not self.history:
            self.history = tokens[1:-1] + [Constants.SEP_WORD] + self.tokenizer.tokenize(pred)
        else:
            self.history = self.history + [Constants.SEP_WORD] + tokens[1:-1] + [Constants.SEP_WORD] + self.tokenizer.tokenize(pred)
        
        words = pred.split(' ')
        for i in range(len(words)):
            if words[i].startswith('[') and words[i].endswith(']') and words[i] != '[UNK]':
                domain, key = words[i][1:-1].split('_')
                if key == 'reference':
                    key = 'Ref'
                elif key == 'trainid':
                    key = 'trainID'
                elif key == 'leaveat':
                    key = 'leaveAt'
                elif key == 'arriveby':
                    key = 'arriveBy'
                elif key == 'price' and domain != 'train':
                    key = 'pricerange'
                elif domain == 'value':
                    if key == 'place':
                        if 'arrive' in pred or 'to' in pred or 'arriving' in pred:
                            key = 'destination'
                        elif 'leave' in pred or 'from' in pred or 'leaving' in pred:
                            key = 'departure'
                    elif key == 'time':
                        if 'arrive' in pred or 'arrival' in pred or 'arriving' in pred:
                            key = 'arriveBy'
                        elif 'leave' in pred or 'departure' in pred or 'leaving' in pred:
                            key = 'leaveAt'
                    elif key == 'count':
                        if 'minute' in pred:
                            key = 'duration'
                        elif 'star' in pred:
                            key = 'stars'
                if key in kb and (domain == kb['domain'] or domain == 'value'):
                    words[i] = kb[key]
                elif domain in self.kbs and key in self.kbs[domain]:
                    words[i] = self.kbs[domain][key]
                else:
                    if domain == 'hospital':
                        if key == 'phone':
                            words[i] = '01223216297'
                        elif key == 'department':
                            words[i] = 'neurosciences critical care unit'
                        elif key == 'address':
                            words[i] = "Lincoln street"
                    elif domain == 'police':
                        if key == 'phone':
                            words[i] = '01223358966'
                        elif key == 'name':
                            words[i] = 'Parkside Police Station'
                        elif key == 'address':
                            words[i] = 'Parkside, Cambridge'
                        elif key == 'postcode':
                            words[i] = '533420'
                    elif domain == 'taxi':
                        if key == 'phone':
                            words[i] = '01223358966'
                        elif key == 'color':
                            words[i] = 'white'
                        elif key == 'type':
                            words[i] = 'toyota'
        sentence = denormalize(" ".join(words))
        
        return sentence

