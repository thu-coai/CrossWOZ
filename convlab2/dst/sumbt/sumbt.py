import collections
import csv
import json
import logging
import os.path
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss

from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.modeling import BertPreTrainedModel
from pytorch_pretrained_bert.tokenization import BertTokenizer

from convlab2.dst.dst import DST
from convlab2.dst.sumbt.config.config import *


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class BertForUtteranceEncoding(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForUtteranceEncoding, self).__init__(config)

        self.config = config
        self.bert = BertModel(config)

    def forward(self, input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False):
        return self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers)


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

        self.scores = None

    def attention(self, q, k, v, d_k, mask=None, dropout=None):

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)

        if dropout is not None:
            scores = dropout(scores)

        self.scores = scores
        output = torch.matmul(scores, v)
        return output

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output

    def get_scores(self):
        return self.scores


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
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


class BeliefTracker(nn.Module):
    def __init__(self):
        super(BeliefTracker, self).__init__()

        self.hidden_dim = HIDDEN_DIM
        self.rnn_num_layers = RNN_NUM_LAYERS
        self.zero_init_rnn = ZERO_INIT_RNN
        self.max_seq_length = MAX_SEQ_LENGTH
        self.max_label_length = MAX_LABEL_LENGTH

        self.attn_head = ATTN_HEAD
        self.device = DEVICE

        if BERT_DIR == '':
            raise ValueError('set BERT-DIR first')

    def init_session(self, num_labels):
        self.num_labels = num_labels
        self.num_slots = len(num_labels)

        ### Utterance Encoder
        self.utterance_encoder = BertForUtteranceEncoding.from_pretrained(
            os.path.join(BERT_DIR)
        )
        self.bert_output_dim = self.utterance_encoder.config.hidden_size
        self.hidden_dropout_prob = self.utterance_encoder.config.hidden_dropout_prob
        if FIX_UTTERANCE_ENCODER:
            for p in self.utterance_encoder.bert.pooler.parameters():
                p.requires_grad = False

        ### slot, slot-value Encoder (not trainable)
        self.sv_encoder = BertForUtteranceEncoding.from_pretrained(
            os.path.join(BERT_DIR))
        for p in self.sv_encoder.bert.parameters():
            p.requires_grad = False

        self.slot_lookup = nn.Embedding(self.num_slots, self.bert_output_dim)
        self.value_lookup = nn.ModuleList([nn.Embedding(num_label, self.bert_output_dim) for num_label in num_labels])

        ### Attention layer
        self.attn = MultiHeadAttention(self.attn_head, self.bert_output_dim, dropout=0)

        ### RNN Belief DST
        self.nbt = None
        if TASK_NAME.find("gru") != -1:
            self.nbt = nn.GRU(input_size=self.bert_output_dim,
                              hidden_size=self.hidden_dim,
                              num_layers=self.rnn_num_layers,
                              dropout=self.hidden_dropout_prob,
                              batch_first=True)
            self.init_parameter(self.nbt)
        elif TASK_NAME.find("lstm") != -1:
            self.nbt = nn.LSTM(input_size=self.bert_output_dim,
                               hidden_size=self.hidden_dim,
                               num_layers=self.rnn_num_layers,
                               dropout=self.hidden_dropout_prob,
                               batch_first=True)
            self.init_parameter(self.nbt)
        if not self.zero_init_rnn:
            self.rnn_init_linear = nn.Sequential(
                nn.Linear(self.bert_output_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.hidden_dropout_prob)
            )

        self.linear = nn.Linear(self.hidden_dim, self.bert_output_dim)
        self.layer_norm = nn.LayerNorm(self.bert_output_dim)

        ### Measure
        self.distance_metric = DISTANCE_METRIC
        if self.distance_metric == "cosine":
            self.metric = torch.nn.CosineSimilarity(dim=-1, eps=1e-08)
        elif self.distance_metric == "euclidean":
            self.metric = torch.nn.PairwiseDistance(p=2.0, eps=1e-06, keepdim=False)

        ### Classifier
        self.nll = CrossEntropyLoss(ignore_index=-1)

        ### Etc.
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

    def initialize_slot_value_lookup(self, label_ids, slot_ids):

        self.sv_encoder.eval()

        # Slot encoding
        slot_type_ids = torch.zeros(slot_ids.size(), dtype=torch.long).to(self.device)
        slot_mask = slot_ids > 0
        hid_slot, _ = self.sv_encoder(slot_ids.view(-1, self.max_label_length),
                                      slot_type_ids.view(-1, self.max_label_length),
                                      slot_mask.view(-1, self.max_label_length),
                                      output_all_encoded_layers=False)
        hid_slot = hid_slot[:, 0, :]
        hid_slot = hid_slot.detach()
        self.slot_lookup = nn.Embedding.from_pretrained(hid_slot, freeze=True)

        for s, label_id in enumerate(label_ids):
            label_type_ids = torch.zeros(label_id.size(), dtype=torch.long).to(self.device)
            label_mask = label_id > 0
            hid_label, _ = self.sv_encoder(label_id.view(-1, self.max_label_length),
                                           label_type_ids.view(-1, self.max_label_length),
                                           label_mask.view(-1, self.max_label_length),
                                           output_all_encoded_layers=False)
            hid_label = hid_label[:, 0, :]
            hid_label = hid_label.detach()
            self.value_lookup[s] = nn.Embedding.from_pretrained(hid_label, freeze=True)
            self.value_lookup[s].padding_idx = -1

        print("Complete initialization of slot and value lookup")

    def _make_aux_tensors(self, ids, len):
        token_type_ids = torch.zeros(ids.size(), dtype=torch.long).to(self.device)
        for i in range(len.size(0)):
            for j in range(len.size(1)):
                if len[i, j, 0] == 0:  # padding
                    break
                elif len[i, j, 1] > 0:  # escape only text_a case
                    start = len[i, j, 0]
                    ending = len[i, j, 0] + len[i, j, 1]
                    token_type_ids[i, j, start:ending] = 1
        attention_mask = ids > 0
        return token_type_ids, attention_mask

    def forward(self, input_ids, input_len, labels, n_gpu=1, target_slot=None):

        # if target_slot is not specified, output values corresponding all slot-types
        if target_slot is None:
            self.target_slot = list(range(0, self.num_slots))
        else:
            raise NotImplementedError()

        ds = input_ids.size(0)  # dialog size
        ts = input_ids.size(1)  # turn size
        bs = ds * ts
        slot_dim = len(self.target_slot)

        # Utterance encoding
        token_type_ids, attention_mask = self._make_aux_tensors(input_ids, input_len)

        hidden, _ = self.utterance_encoder(input_ids.view(-1, self.max_seq_length),
                                           token_type_ids.view(-1, self.max_seq_length),
                                           attention_mask.view(-1, self.max_seq_length),
                                           output_all_encoded_layers=False)
        hidden = torch.mul(hidden, attention_mask.view(-1, self.max_seq_length, 1).expand(hidden.size()).float())
        hidden = hidden.repeat(slot_dim, 1, 1)  # [(slot_dim*ds*ts), bert_seq, hid_size]

        hid_slot = self.slot_lookup.weight[self.target_slot, :]  # Select target slot embedding
        hid_slot = hid_slot.repeat(1, bs).view(bs * slot_dim, -1)  # [(slot_dim*ds*ts), bert_seq, hid_size]

        # Attended utterance vector
        hidden = self.attn(hid_slot, hidden, hidden,
                           mask=attention_mask.view(-1, 1, self.max_seq_length).repeat(slot_dim, 1, 1))
        hidden = hidden.squeeze()  # [slot_dim*ds*ts, bert_dim]
        hidden = hidden.view(slot_dim, ds, ts, -1).view(-1, ts, self.bert_output_dim)

        # NBT
        if self.zero_init_rnn:
            h = torch.zeros(self.rnn_num_layers, input_ids.shape[0] * slot_dim, self.hidden_dim).to(
                self.device)  # [1, slot_dim*ds, hidden]
        else:
            h = hidden[:, 0, :].unsqueeze(0).repeat(self.rnn_num_layers, 1, 1)
            h = self.rnn_init_linear(h)

        if isinstance(self.nbt, nn.GRU):
            rnn_out, _ = self.nbt(hidden, h)  # [slot_dim*ds, turn, hidden]
        elif isinstance(self.nbt, nn.LSTM):
            c = torch.zeros(self.rnn_num_layers, input_ids.shape[0] * slot_dim, self.hidden_dim).to(
                self.device)  # [1, slot_dim*ds, hidden]
            rnn_out, _ = self.nbt(hidden, (h, c))  # [slot_dim*ds, turn, hidden]
        rnn_out = self.layer_norm(self.linear(self.dropout(rnn_out)))

        hidden = rnn_out.view(slot_dim, ds, ts, -1)

        # Label (slot-value) encoding
        loss = 0
        loss_slot = []
        pred_slot = []
        output = []
        for s, slot_id in enumerate(self.target_slot):  ## note: target_slots are successive
            # loss calculation
            hid_label = self.value_lookup[slot_id].weight
            num_slot_labels = hid_label.size(0)

            _hid_label = hid_label.unsqueeze(0).unsqueeze(0).repeat(ds, ts, 1, 1).view(ds * ts * num_slot_labels, -1)
            _hidden = hidden[s, :, :, :].unsqueeze(2).repeat(1, 1, num_slot_labels, 1).view(ds * ts * num_slot_labels,
                                                                                            -1)
            _dist = self.metric(_hid_label, _hidden).view(ds, ts, num_slot_labels)

            if self.distance_metric == "euclidean":
                _dist = -_dist
            _, pred = torch.max(_dist, -1)
            pred_slot.append(pred.view(ds, ts, 1))
            output.append(_dist)

            if labels is not None:
                _loss = self.nll(_dist.view(ds * ts, -1), labels[:, :, s].view(-1))
                loss_slot.append(_loss.item())
                loss += _loss

        if labels is None:
            return output

        # calculate joint accuracy
        pred_slot = torch.cat(pred_slot, 2)  # [ds, ts, slot_dim]
        accuracy = (pred_slot == labels).view(-1, slot_dim)
        acc_slot = torch.sum(accuracy, 0).float() \
                   / torch.sum(labels.view(-1, slot_dim) > -1, 0).float()
        acc = sum(torch.sum(accuracy, 1) / slot_dim).float() \
              / torch.sum(labels[:, :, 0].view(-1) > -1, 0).float()  # joint accuracy

        if n_gpu == 1:
            return loss, loss_slot, acc, acc_slot, pred_slot
        else:
            return loss.unsqueeze(0), None, acc.unsqueeze(0), acc_slot.unsqueeze(0), pred_slot.unsqueeze(0)

    @staticmethod
    def init_parameter(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            torch.nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.GRU) or isinstance(module, nn.LSTM):
            torch.nn.init.xavier_normal_(module.weight_ih_l0)
            torch.nn.init.xavier_normal_(module.weight_hh_l0)
            torch.nn.init.constant_(module.bias_ih_l0, 0.0)
            torch.nn.init.constant_(module.bias_hh_l0, 0.0)


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, max_turn_length):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = [{label: i for i, label in enumerate(labels)} for labels in label_list]
    slot_dim = len(label_list)

    features = []
    prev_dialogue_idx = None
    all_padding = [0] * max_seq_length
    all_padding_len = [0, 0]

    max_turn = 0
    for (ex_index, example) in enumerate(examples):
        if max_turn < int(example.guid.split('-')[2]):
            max_turn = int(example.guid.split('-')[2])
    max_turn_length = min(max_turn + 1, max_turn_length)
    logger.info("max_turn_length = %d" % max_turn)

    for (ex_index, example) in enumerate(examples):
        tokens_a = [x if x != '#' else '[SEP]' for x in tokenizer.tokenize(example.text_a)]
        tokens_b = None
        if example.text_b:
            tokens_b = [x if x != '#' else '[SEP]' for x in tokenizer.tokenize(example.text_b)]
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        input_len = [len(tokens), 0]

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            input_len[1] = len(tokens_b) + 1

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        assert len(input_ids) == max_seq_length

        FLAG_TEST = False
        if example.label is not None:
            label_id = []
            label_info = 'label: '
            for i, label in enumerate(example.label):
                if label == 'dontcare':
                    label = 'do not care'
                label_id.append(label_map[i][label])
                label_info += '%s (id = %d) ' % (label, label_map[i][label])

            if ex_index < 5:
                logger.info("*** Example ***")
                logger.info("guid: %s" % example.guid)
                logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("input_len: %s" % " ".join([str(x) for x in input_len]))
                logger.info("label: " + label_info)
        else:
            FLAG_TEST = True
            label_id = None

        curr_dialogue_idx = example.guid.split('-')[1]
        curr_turn_idx = int(example.guid.split('-')[2])

        if prev_dialogue_idx is not None and prev_dialogue_idx != curr_dialogue_idx:
            if prev_turn_idx < max_turn_length:
                features += [InputFeatures(input_ids=all_padding,
                                           input_len=all_padding_len,
                                           label_id=[-1] * slot_dim)] \
                            * (max_turn_length - prev_turn_idx - 1)
            assert len(features) % max_turn_length == 0

        if prev_dialogue_idx is None or prev_turn_idx < max_turn_length:
            features.append(
                InputFeatures(input_ids=input_ids,
                              input_len=input_len,
                              label_id=label_id))

        prev_dialogue_idx = curr_dialogue_idx
        prev_turn_idx = curr_turn_idx

    if prev_turn_idx < max_turn_length:
        features += [InputFeatures(input_ids=all_padding,
                                   input_len=all_padding_len,
                                   label_id=[-1] * slot_dim)] \
                    * (max_turn_length - prev_turn_idx - 1)
    assert len(features) % max_turn_length == 0

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_len = torch.tensor([f.input_len for f in features], dtype=torch.long)
    if not FLAG_TEST:
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    # reshape tensors to [#batch, #max_turn_length, #max_seq_length]
    all_input_ids = all_input_ids.view(-1, max_turn_length, max_seq_length)
    all_input_len = all_input_len.view(-1, max_turn_length, 2)
    if not FLAG_TEST:
        all_label_ids = all_label_ids.view(-1, max_turn_length, slot_dim)
    else:
        all_label_ids = None

    return all_input_ids, all_input_len, all_label_ids


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
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
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_len, label_id):
        self.input_ids = input_ids
        self.input_len = input_len
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if len(line) > 0 and line[0][0] == '#':  # ignore comments (starting with '#')
                    continue
                lines.append(line)
            return lines


class Processor(DataProcessor):
    """Processor for the belief tracking dataset (GLUE version)."""

    def __init__(self):
        super(Processor, self).__init__()

        # WOZ2.0 dataset
        if DATASET == 'woz':
            fp_ontology = open(os.path.join(DATA_DIR, "ontology_dstc2_en.json"), "r")
            ontology = json.load(fp_ontology)
            ontology = ontology["informable"]
            del ontology["request"]
            for slot in ontology.keys():
                ontology[slot].append("do not care")
                ontology[slot].append("none")
            fp_ontology.close()

        # MultiWOZ dataset
        elif DATASET == 'multiwoz':
            fp_ontology = open(os.path.join(DATA_DIR, "ontology.json"), "r")
            ontology = json.load(fp_ontology)
            for slot in ontology.keys():
                ontology[slot].append("none")
            fp_ontology.close()

            if not TARGET_SLOT == 'all':
                slot_idx = {'attraction': '0:1:2', 'bus': '3:4:5:6', 'hospital': '7',
                            'hotel': '8:9:10:11:12:13:14:15:16:17',
                            'restaurant': '18:19:20:21:22:23:24', 'taxi': '25:26:27:28', 'train': '29:30:31:32:33:34'}
                target_slot = []
                for key, value in slot_idx.items():
                    if key != TARGET_SLOT:
                        target_slot.append(value)
                partial_target_slot = ':'.join(target_slot)

        # sorting the ontology according to the alphabetic order of the slots
        ontology = collections.OrderedDict(sorted(ontology.items()))

        # select slots to train
        nslots = len(ontology.keys())
        target_slot = list(ontology.keys())
        if TARGET_SLOT == 'all':
            self.target_slot_idx = [*range(0, nslots)]
        else:
            self.target_slot_idx = sorted([int(x) for x in partial_target_slot.split(':')])

        for idx in range(0, nslots):
            if not idx in self.target_slot_idx:
                del ontology[target_slot[idx]]

        self.ontology = ontology
        self.target_slot = list(self.ontology.keys())
        for i, slot in enumerate(self.target_slot):
            if slot == "pricerange":
                self.target_slot[i] = "price range"

        logger.info('Processor: target_slot')
        logger.info(self.target_slot)

    def get_train_examples(self, data_dir, accumulation=False):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", accumulation)

    def get_dev_examples(self, data_dir, accumulation=False):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev", accumulation)

    def get_test_examples(self, data_dir, accumulation=False):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test", accumulation)

    def get_labels(self):
        """See base class."""
        return [self.ontology[slot] for slot in self.target_slot]

    def _create_examples(self, lines, set_type, accumulation=False):
        """Creates examples for the training and dev sets."""
        prev_dialogue_index = None
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s-%s" % (set_type, line[0], line[1])  # line[0]: dialogue index, line[1]: turn index
            if accumulation:
                if prev_dialogue_index is None or prev_dialogue_index != line[0]:
                    text_a = line[2]
                    text_b = line[3]
                    prev_dialogue_index = line[0]
                else:
                    # The symbol '#' will be replaced with '[SEP]' after tokenization.
                    text_a = line[2] + " # " + text_a
                    text_b = line[3] + " # " + text_b
            else:
                text_a = line[2]  # line[2]: user utterance
                text_b = line[3]  # line[3]: system response

            label = [line[4 + idx] for idx in self.target_slot_idx]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class SUMBTTracker(DST):
    def __init__(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

    def update(self, action):
        return self.update_batch([action])[0]

    def add_track(self, dialog_history):
        self.current_turn = 0
        # dialog_history: [{'sys': ..., 'usr': ...}, ...]
        examples = []

        for turn_idx, turn in enumerate(dialog_history):
            guid = "%s-%s-%s" % ('track', 1, turn_idx)  # line[0]: dialogue index, line[1]: turn index
            if self.accumulation:
                if turn_idx == 0:
                    assert turn['sys'] == ''
                    text_a = turn['usr']
                    text_b = ''
                else:
                    # The symbol '#' will be replaced with '[SEP]' after tokenization.
                    text_a = turn['usr'] + " # " + text_a
                    text_b = turn['sys'] + " # " + text_b
            else:
                text_a = turn['usr']  # line[2]: user utterance
                text_b = turn['sys']  # line[3]: system response

            # label = ['none' for idx in self.target_slot]
            label = None

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        all_input_ids, all_input_len, all_label_ids = convert_examples_to_features(
            examples, self.label_list, MAX_SEQ_LENGTH, self.tokenizer, MAX_TURN_LENGTH)
        all_input_ids, all_input_len = all_input_ids.to(DEVICE), all_input_len.to(
            DEVICE)

        pred_output = self.belief_tracker(all_input_ids, all_input_len, None, N_GPU)  # [num_slot, ds, num_turn, num_slot_value]
        pred_output = [torch.argmax(pred_slot_res, 2) for pred_slot_res in pred_output]  # [num_slot, ds, num_turn]

        self.pred_slot = []  # [[[(slot, value), slot_value2, ...], turn2, ...], dialog2, ...]
        for slot_idx, slot_pred_res in enumerate(pred_output):
            slot_str = self.processor.target_slot[slot_idx]
            ds, num_turn = slot_pred_res.shape
            if self.pred_slot == []:
                for d in range(ds):
                    self.pred_slot.append([])
                for t in range(num_turn):
                    for d in range(ds):
                        self.pred_slot[d].append([])
                for d in range(ds):
                    for t in range(num_turn):
                        self.pred_slot[d][t] = {}

            for d in range(ds):
                for t in range(num_turn):
                    slot_value_idx = slot_pred_res[d, t]
                    pred_slot_value = self.processor.ontology[slot_str][slot_value_idx]
                    self.pred_slot[d][t][slot_str] = pred_slot_value

    def update_batch(self, batch_action=None):
        ret = [d[self.current_turn] for d in self.pred_slot]
        return ret


if __name__ == '__main__':
    tracker = SUMBTTracker()
    tracker.load_weights()
    dialog_history = [
        {'usr': "I would like a taxi from Saint John 's college to Pizza Hut Fen Ditton .", 'sys': ''},
        {'sys': "What time do you want to leave and what time do you want to arrive by ?",
         'usr': "I want to leave after 17:15 ."},
        {'sys': "Booking completed ! your taxi will be blue honda Contact number is 07218068540",
         'usr': "Thank you for all the help ! I appreciate it ."}
    ]
    tracker.add_track(dialog_history)
    for i in range(3):
        print(tracker.update_batch())
