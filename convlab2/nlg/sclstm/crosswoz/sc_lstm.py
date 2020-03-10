import configparser
import os
import zipfile
from copy import deepcopy
from collections import defaultdict
from pprint import pprint
import torch
import re

from convlab2.util.file_util import cached_path
from convlab2.nlg.sclstm.crosswoz.loader.dataset_woz import SimpleDatasetWoz
from convlab2.nlg.sclstm.model.lm_deep import LMDeep
from convlab2.nlg.nlg import NLG

DEFAULT_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DEFAULT_ARCHIVE_FILE = os.path.join(DEFAULT_DIRECTORY, "nlg_sclstm_crosswoz.zip")


def parse(is_user):
    if is_user:
        args = {
            'model_path': 'sclstm_usr.pt',
            'n_layer': 1,
            'beam_size': 10
        }
    else:
        args = {
            'model_path': 'sclstm.pt',
            'n_layer': 1,
            'beam_size': 10
        }

    config = configparser.ConfigParser()
    if is_user:
        config.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config/config_usr.cfg'))
    else:
        config.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config/config.cfg'))
    config.set('DATA', 'dir', os.path.dirname(os.path.abspath(__file__)))

    return args, config


class SCLSTM(NLG):
    def __init__(self, 
                 archive_file=DEFAULT_ARCHIVE_FILE, 
                 use_cuda=False,
                 is_user=False,
                 model_file='https://convlab.blob.core.windows.net/convlab-2/nlg_sclstm_crosswoz.zip'):

        if not os.path.isfile(archive_file):
            if not model_file:
                raise Exception("No model for SC-LSTM is specified!")
            archive_file = cached_path(model_file)
        model_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.exists(os.path.join(model_dir, 'resource')):
            archive = zipfile.ZipFile(archive_file, 'r')
            archive.extractall(model_dir)

        self.USE_CUDA = use_cuda
        self.args, self.config = parse(is_user)
        self.dataset = SimpleDatasetWoz(self.config)

        # get model hyper-parameters
        hidden_size = self.config.getint('MODEL', 'hidden_size')

        # get feat size
        d_size = self.dataset.do_size + self.dataset.da_size + self.dataset.sv_size  # len of 1-hot feat
        vocab_size = len(self.dataset.word2index)

        self.model = LMDeep('sclstm', vocab_size, vocab_size, hidden_size, d_size, n_layer=self.args['n_layer'], use_cuda=use_cuda)
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.args['model_path'])
        # print(model_path)
        assert os.path.isfile(model_path)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        for name, param in self.model.named_parameters():
            print(name, param.shape, param.device, param.requires_grad)
        if use_cuda:
            self.model.cuda()

    def generate_delex(self, meta):
        """
        meta = [
                    [
                        "General",
                        "greet",
                        "none",
                        "none"
                    ],
                    [
                        "Request",
                        "景点",
                        "名称",
                        ""
                    ],
                    [
                        "Inform",
                        "景点",
                        "门票",
                        "免费"
                    ]
                ]
        """
        intent_list = []
        intent_frequency = defaultdict(int)
        feat_dict = dict()
        for act in meta:
            cur_act = deepcopy(act)

            # intent list
            facility = None  # for 酒店设施
            if '酒店设施' in cur_act[2]:
                facility = cur_act[2].split('-')[1]
                if cur_act[0] == 'Inform':
                    cur_act[2] = cur_act[2].split('-')[0] + '+' + cur_act[3]
                elif cur_act[0] == 'Request':
                    cur_act[2] = cur_act[2].split('-')[0]
            if cur_act[0] == 'Select':
                cur_act[2] = '源领域+' + cur_act[3]
            intent = '+'.join(cur_act[:-1])
            if '+'.join(cur_act) == 'Inform+景点+门票+免费' or str(cur_act[-1]) == '无':
                intent = '+'.join(cur_act)
            intent_list.append(intent)

            intent_frequency[intent] += 1

            # content replacement
            value = 'none'
            freq = 'none'
            if (act[0] in ['Inform', 'Recommend'] or '酒店设施' in intent) and not intent.endswith('无'):
                if '酒店设施' in intent:
                    value = facility
                else:
                    value = act[3]
                    freq = str(intent_frequency[intent])
            elif act[0] == 'Request':
                freq = '?'
                value = '?'
            elif act[0] == 'Select':
                value = act[3]

            # generate the formation in feat.json
            new_act = intent.split('+')
            if new_act[0] == 'General':
                feat_key = new_act[0] + '-' + new_act[1]
            else:
                feat_key = new_act[1] + '-' + new_act[0]
            if new_act[2] == '酒店设施' and new_act[0] == 'Inform':
                try:
                    feat_value = [new_act[2] + '+' + new_act[3], freq, value]
                except:
                    print(new_act)
            elif intent.endswith('无'):
                feat_value = [new_act[2] + '+无', freq, value]
            elif intent.endswith('免费'):
                feat_value = [new_act[2] + '+免费', freq, value]
            else:
                feat_value = [new_act[2], freq, value]
            feat_dict[feat_key] = feat_dict.get(feat_key, [])
            feat_dict[feat_key].append(feat_value)

        meta = deepcopy(feat_dict)

        # remove invalid dialog act
        meta_ = deepcopy(meta)
        for k, v in meta.items():
            for triple in v:
                voc = 'd-a-s-v:' + k + '-' + triple[0] + '-' + triple[1]
                if voc not in self.dataset.cardinality:
                    meta_[k].remove(triple)
            if not meta_[k]:
                del (meta_[k])
        meta = meta_

        # mapping the inputs
        do_idx, da_idx, sv_idx, featStr = self.dataset.getFeatIdx(meta)
        do_cond = [1 if i in do_idx else 0 for i in range(self.dataset.do_size)]  # domain condition
        da_cond = [1 if i in da_idx else 0 for i in range(self.dataset.da_size)]  # dial act condition
        sv_cond = [1 if i in sv_idx else 0 for i in range(self.dataset.sv_size)]  # slot/value condition
        feats = [do_cond + da_cond + sv_cond]

        feats_var = torch.FloatTensor(feats)
        if self.USE_CUDA:
            feats_var = feats_var.cuda()

        decoded_words = self.model.generate(self.dataset, feats_var, self.args['beam_size'])
        delex = decoded_words[0]  # (beam_size)

        return delex

    def generate_slots(self, meta):
        meta = deepcopy(meta)
        
        delex = self.generate_delex(meta)
        # get all informable or requestable slots
        slots = []
        for sen in delex:
            slot = []
            counter = {}
            words = sen.split()
            for word in words:
                if word.startswith('slot-'):
                    placeholder = word[5:]
                    if placeholder not in counter:
                        counter[placeholder] = 1
                    else:
                        counter[placeholder] += 1
                    slot.append(placeholder+'-'+str(counter[placeholder]))
            slots.append(slot)
            
        # for i in range(self.args.beam_size):
        #     print(i, slots[i])

        return slots[0]

    def _value_replace(self, sentences, dialog_act):
        ori_sen = deepcopy(sentences)
        dialog_act = deepcopy(dialog_act)
        intent_frequency = defaultdict(int)
        for act in dialog_act:
            intent = self._prepare_intent_string(deepcopy(act))
            intent_frequency[intent] += 1
            if intent_frequency[intent] > 1:  # if multiple same intents...
                intent += str(intent_frequency[intent])

            if '酒店设施' in intent:
                try:
                    sentences = sentences.replace('[' + intent + ']', act[2].split('-')[1])
                    sentences = sentences.replace('[' + intent + '1]', act[2].split('-')[1])
                except Exception as e:
                    print('Act causing problem in replacement:')
                    pprint(act)
                    raise e
            if act[0] == 'Inform' and act[3] == "无":
                sentences = sentences.replace('[主体]', act[1])
                sentences = sentences.replace('[属性]', act[2])
            sentences = sentences.replace('[' + intent + ']', act[3])
            sentences = sentences.replace('[' + intent + '1]', act[3])  # if multiple same intents and this is 1st

        # if '[' in sentences and ']' in sentences:
        #     print('\n\nValue replacement not completed!!! Current sentence: %s' % sentences)
        #     print('Current DA:')
        #     pprint(dialog_act)
        #     print('ori sen', ori_sen)
        #     pattern = re.compile(r'(\[[^\[^\]]+\])')
        #     slots = pattern.findall(sentences)
        #     for slot in slots:
        #         sentences = sentences.replace(slot, ' ')
        #     print('after replace:', sentences)
            # raise Exception
        return sentences

    def _prepare_intent_string(self, cur_act):
        """
        Generate the intent form **to be used in selecting templates** (rather than value replacement)
        :param cur_act: one act list
        :return: one intent string
        """
        cur_act = deepcopy(cur_act)
        if cur_act[0] == 'Inform' and '酒店设施' in cur_act[2]:
            cur_act[2] = cur_act[2].split('-')[0] + '+' + cur_act[3]
        elif cur_act[0] == 'Request' and '酒店设施' in cur_act[2]:
            cur_act[2] = cur_act[2].split('-')[0]
        if cur_act[0] == 'Select':
            cur_act[2] = '源领域+' + cur_act[3]
        try:
            if '+'.join(cur_act) == 'Inform+景点+门票+免费':
                intent = '+'.join(cur_act)
            # "Inform+景点+周边酒店+无"
            elif cur_act[3] == '无':
                intent = '+'.join(cur_act)
            else:
                intent = '+'.join(cur_act[:-1])
        except Exception as e:
            print('Act causing error:')
            pprint(cur_act)
            raise e
        return intent
    
    def generate(self, meta):
        meta = [[str(x[0]), str(x[1]), str(x[2]), str(x[3]).lower()] for x in meta]
        meta = deepcopy(meta)
        
        delex = self.generate_delex(meta)

        return self._value_replace(delex[0].replace('UNK_token', '').replace(' ', ''), meta)


if __name__ == '__main__':
    model_sys = SCLSTM(is_user=True, use_cuda=True)
    print(model_sys.generate([['Inform', '餐馆', '人均消费', '100-150元'], ['Request', '餐馆', '电话', '']]))
