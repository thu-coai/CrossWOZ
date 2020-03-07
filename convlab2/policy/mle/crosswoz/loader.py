import os
import json
import pickle
import zipfile
import torch
import torch.utils.data as data
from convlab2.util.crosswoz.state import default_state
from convlab2.dst.rule.crosswoz.dst import RuleDST
from convlab2.policy.vector.vector_crosswoz import CrossWozVector
from copy import deepcopy


class PolicyDataLoaderCrossWoz():

    def __init__(self):
        root_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
        voc_file = os.path.join(root_dir, 'data/crosswoz/sys_da_voc.json')
        voc_opp_file = os.path.join(root_dir, 'data/crosswoz/usr_da_voc.json')
        self.vector = CrossWozVector(sys_da_voc_json=voc_file, usr_da_voc_json=voc_opp_file)

        processed_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed_data')
        if os.path.exists(processed_dir):
            print('Load processed data file')
            self._load_data(processed_dir)
        else:
            print('Start preprocessing the dataset')
            self._build_data(root_dir, processed_dir)

    def _build_data(self, root_dir, processed_dir):
        raw_data = {}
        for part in ['train', 'val', 'test']:
            archive = zipfile.ZipFile(os.path.join(root_dir, 'data/crosswoz/{}.json.zip'.format(part)), 'r')
            with archive.open('{}.json'.format(part), 'r') as f:
                raw_data[part] = json.load(f)

        self.data = {}
        # for cur domain update
        dst = RuleDST()
        for part in ['train', 'val', 'test']:
            self.data[part] = []

            for key in raw_data[part]:
                sess = raw_data[part][key]['messages']
                dst.init_session()
                for i, turn in enumerate(sess):
                    if turn['role'] == 'usr':
                        dst.update(usr_da=turn['dialog_act'])
                        if i + 2 == len(sess):
                            dst.state['terminated'] = True
                    else:
                        for domain, svs in turn['sys_state'].items():
                            for slot, value in svs.items():
                                if slot != 'selectedResults':
                                    dst.state['belief_state'][domain][slot] = value
                        action = turn['dialog_act']
                        self.data[part].append([self.vector.state_vectorize(deepcopy(dst.state)),
                                                self.vector.action_vectorize(action)])
                        dst.state['system_action'] = turn['dialog_act']

        os.makedirs(processed_dir)
        for part in ['train', 'val', 'test']:
            with open(os.path.join(processed_dir, '{}.pkl'.format(part)), 'wb') as f:
                pickle.dump(self.data[part], f)

    def _load_data(self, processed_dir):
        self.data = {}
        for part in ['train', 'val', 'test']:
            with open(os.path.join(processed_dir, '{}.pkl'.format(part)), 'rb') as f:
                self.data[part] = pickle.load(f)

    def create_dataset(self, part, batchsz):
        print('Start creating {} dataset'.format(part))
        s = []
        a = []
        for item in self.data[part]:
            s.append(torch.Tensor(item[0]))
            a.append(torch.Tensor(item[1]))
        s = torch.stack(s)
        a = torch.stack(a)
        dataset = Dataset(s, a)
        dataloader = data.DataLoader(dataset, batchsz, True)
        print('Finish creating {} dataset'.format(part))
        return dataloader


class Dataset(data.Dataset):
    def __init__(self, s_s, a_s):
        self.s_s = s_s
        self.a_s = a_s
        self.num_total = len(s_s)

    def __getitem__(self, index):
        s = self.s_s[index]
        a = self.a_s[index]
        return s, a

    def __len__(self):
        return self.num_total

