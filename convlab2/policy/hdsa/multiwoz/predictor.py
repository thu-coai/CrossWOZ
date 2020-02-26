import os
import zipfile
import torch

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.tokenization import BertTokenizer

from convlab2.policy.hdsa.multiwoz.transformer import Constants
from convlab2.util.file_util import cached_path
from convlab2.util.multiwoz.dbquery import Database

def examine(domain, slot):
    if slot == "addr":
        slot = 'address'
    elif slot == "post":
        slot = 'postcode'
    elif slot == "ref":
        slot = 'ref'
    elif slot == "car":
        slot = "type"
    elif slot == 'dest':
        slot = 'destination'
    elif domain == 'train' and slot == 'id':
        slot = 'trainid'
    elif slot == 'leave':
        slot = 'leaveat'
    elif slot == 'arrive':
        slot = 'arriveby'
    elif slot == 'price':
        slot = 'pricerange'
    elif slot == 'depart':
        slot = 'departure'
    elif slot == 'name':
        slot = 'name'
    elif slot == 'type':
        slot = 'type'
    elif slot == 'area':
        slot = 'area'
    elif slot == 'parking':
        slot = 'parking'
    elif slot == 'internet':
        slot = 'internet'
    elif slot == 'stars':
        slot = 'stars'
    elif slot == 'food':
        slot = 'food'
    elif slot == 'phone':
        slot = 'phone'
    elif slot == 'day':
        slot = 'day'
    else:
        slot = 'illegal'
    return slot

def truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, file, turn, guid, text_m, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.file = file
        self.turn = turn
        self.guid = guid
        self.text_m = text_m
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, file, turn, input_ids, input_mask, segment_ids, label_id):
        self.file = file
        self.turn = turn
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

class HDSA_predictor():
    def __init__(self, archive_file, model_file=None, use_cuda=False):
        if not os.path.isfile(archive_file):
            if not model_file:
              raise Exception("No model for DA-predictor is specified!")
        archive_file = cached_path(model_file)
        model_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.exists(os.path.join(model_dir, 'checkpoints')):
          archive = zipfile.ZipFile(archive_file, 'r')
          archive.extractall(model_dir)
        
        
        load_dir = os.path.join(model_dir, "checkpoints/predictor/save_step_15120")
        self.db=Database()
        if not os.path.exists(load_dir):
            archive = zipfile.ZipFile(f'{load_dir}.zip', 'r')
            archive.extractall(os.path.dirname(load_dir))
        
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        self.max_seq_length = 256
        self.domain = 'restaurant'
        self.model = BertForSequenceClassification.from_pretrained(load_dir, 
            cache_dir=os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(-1)), num_labels=44)
        self.device = 'cuda' if use_cuda else 'cpu'
        self.model.to(self.device)
        
    def gen_example(self, state):
        file = ''
        turn = 0
        guid = 'infer'
        
        act = state['user_action']
        for w in act:
            d, f = w.split('-')
            if Constants.domains.index(d.lower()) < 8:
                self.domain = d.lower()
        hierarchical_act_vecs = [0 for _ in range(44)] # fake target
        
        meta = state['belief_state']
        constraints = []
        if self.domain != 'bus':
            for slot in meta[self.domain]['semi']:
                if meta[self.domain]['semi'][slot] != "":
                    constraints.append([slot, meta[self.domain]['semi'][slot]])
        query_result = self.db.query(self.domain, constraints)
        if not query_result:
            kb = {'count':'0'}
            src = "no information"
        else:
            kb = query_result[0]
            kb['count'] = str(len(query_result))
            src = []
            for k, v in kb.items():
                k = examine(self.domain, k.lower())
                if k != 'illegal' and isinstance(v, str):
                    src.extend([k, 'is', v])
            src = " ".join(src)
        
        usr = state['history'][-1][-1]
        sys = state['history'][-1][-2] if len(state['history'][-1]) > 1 else None
        
        example = InputExample(file, turn, guid, src, usr, sys, hierarchical_act_vecs)
        kb['domain'] = self.domain
        return example, kb

    def gen_feature(self, example):
        tokens_a = self.tokenizer.tokenize(example.text_a)
        tokens_b = self.tokenizer.tokenize(example.text_b)
        tokens_m = self.tokenizer.tokenize(example.text_m)
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        truncate_seq_pair(tokens_a, tokens_b, self.max_seq_length - 3)

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * (len(tokens_a) + 2)

        assert len(tokens) == len(segment_ids)

        tokens += tokens_b + ["[SEP]"]
        segment_ids += [1] * (len(tokens_b) + 1)

        if len(tokens) < self.max_seq_length:
            if len(tokens_m) > self.max_seq_length - len(tokens) - 1:
                tokens_m = tokens_m[:self.max_seq_length - len(tokens) - 1]

            tokens += tokens_m + ['[SEP]']
            segment_ids += [0] * (len(tokens_m) + 1)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)
        # Zero-pad up to the sequence length.
        padding = [0] * (self.max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length

        feature = InputFeatures(file=example.file,
                          turn=example.turn,
                          input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=example.label)
        return feature

    def predict(self, state):
        
        example, kb = self.gen_example(state)
        feature = self.gen_feature(example)
        
        input_ids = torch.tensor([feature.input_ids], dtype=torch.long).to(self.device)
        input_masks = torch.tensor([feature.input_mask], dtype=torch.long).to(self.device)
        segment_ids = torch.tensor([feature.segment_ids], dtype=torch.long).to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids, segment_ids, input_masks, labels=None)
            logits = torch.sigmoid(logits)
        preds = (logits > 0.4).float()
        preds_numpy = preds.cpu().nonzero().squeeze().numpy()
        
#        for i in preds_numpy:
#            if i < 10:
#                print(Constants.domains[i], end=' ')
#            elif i < 17:
#                print(Constants.functions[i-10], end=' ')
#            else:
#                print(Constants.arguments[i-17], end=' ')
#        print()
        
        return preds, kb
