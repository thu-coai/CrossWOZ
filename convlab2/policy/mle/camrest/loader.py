import os
import json
import pickle
import zipfile
from convlab2.util.camrest.state import default_state
from convlab2.util.dataloader.module_dataloader import ActPolicyDataloader
from convlab2.policy.vector.vector_camrest import CamrestVector

class ActPolicyDataLoaderCamrest(ActPolicyDataloader):
    
    def __init__(self):
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
        voc_file = os.path.join(root_dir, 'data/camrest/sys_da_voc.txt')
        voc_opp_file = os.path.join(root_dir, 'data/camrest/usr_da_voc.txt')
        self.vector = CamrestVector(voc_file, voc_opp_file)
        
        processed_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed_data')
        if os.path.exists(processed_dir):
            print('Load processed data file')
            self._load_data(processed_dir)
        else:
            print('Start preprocessing the dataset')
            self._build_data(root_dir, processed_dir)
        
    def _build_data(self, root_dir, processed_dir): # TODO
        raw_data = {}
        for part in ['train', 'val', 'test']:
            archive = zipfile.ZipFile(os.path.join(root_dir, 'data/camrest/{}.json.zip'.format(part)), 'r')
            with archive.open('{}.json'.format(part), 'r') as f:
                raw_data[part] = json.load(f)
        
        self.data = {}
        for part in ['train', 'val', 'test']:
            self.data[part] = []
            
            for key in raw_data[part]:
                sess = key['dial']
                state = default_state()
                action = {}
                for i, turn in enumerate(sess):
                    state['user_action'] = turn['usr']['dialog_act']
                    if i + 1 == len(sess):
                        state['terminated'] = True
                    for da in turn['usr']['slu']:
                        if da['slots'][0][0] != 'slot':
                            state['belief_state'][da['slots'][0][0]] = da['slots'][0][1]
                    action = turn['sys']['dialog_act']
                    self.data[part].append([self.vector.state_vectorize(state),
                         self.vector.action_vectorize(action)])
                    state['system_action'] = turn['sys']['dialog_act']
        
        os.makedirs(processed_dir)
        for part in ['train', 'val', 'test']:
            with open(os.path.join(processed_dir, '{}.pkl'.format(part)), 'wb') as f:
                pickle.dump(self.data[part], f)
