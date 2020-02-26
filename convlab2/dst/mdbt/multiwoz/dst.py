import json
import os
import time
import tensorflow as tf
import shutil
import zipfile

from convlab2.dst.mdbt.mdbt import MDBT
from convlab2.dst.mdbt.mdbt_util import load_word_vectors, load_ontology, load_woz_data_new
from convlab2.util.dataloader.module_dataloader import AgentDSTDataloader
from convlab2.util.dataloader.dataset_dataloader import MultiWOZDataloader
from convlab2.util.file_util import cached_path

train_batch_size = 1
batches_per_eval = 10
no_epochs = 600
device = "gpu"
start_batch = 0


class MultiWozMDBT(MDBT):
    def __init__(self, data_dir='configs', data=None):
        """Constructor of MultiWOzMDBT class.
        Args:
            data_dir (str): The path of data dir, where the root path is tatk/dst/mdbt/multiwoz.
        """
        self.file_url = 'https://tatk-data.s3-ap-northeast-1.amazonaws.com/mdbt_multiwoz_sys.zip'
        local_path = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(local_path, data_dir)  # abstract data path

        self.validation_url = os.path.join(self.data_dir, 'data/validate.json')
        self.training_url = os.path.join(self.data_dir, 'data/train.json')
        self.testing_url = os.path.join(self.data_dir, 'data/test.json')

        self.word_vectors_url = os.path.join(self.data_dir, 'word-vectors/paragram_300_sl999.txt')
        self.ontology_url = os.path.join(self.data_dir, 'data/ontology.json')
        self.model_url = os.path.join(self.data_dir, 'models/model-1')
        self.graph_url = os.path.join(self.data_dir, 'graphs/graph-1')
        self.results_url = os.path.join(self.data_dir, 'results/log-1.txt')
        self.kb_url = os.path.join(self.data_dir, 'data/')  # not used
        self.train_model_url = os.path.join(self.data_dir, 'train_models/model-1')
        self.train_graph_url = os.path.join(self.data_dir, 'train_graph/graph-1')

        self.auto_download()

        print('Configuring MDBT model...')
        self.word_vectors = load_word_vectors(self.word_vectors_url)

        # Load the ontology and extract the feature vectors
        self.ontology, self.ontology_vectors, self.slots = load_ontology(self.ontology_url, self.word_vectors)

        # Load and process the training data
        self.test_dialogues, self.actual_dialogues = load_woz_data_new(data['test'], self.word_vectors,
                                                                   self.ontology, url=self.testing_url)
        self.no_dialogues = len(self.test_dialogues)

        super(MultiWozMDBT, self).__init__(self.ontology_vectors, self.ontology, self.slots, self.data_dir)

    def auto_download(self):
        """Automatically download the pretrained model and necessary data."""
        if os.path.exists(os.path.join(self.data_dir, 'models')) and \
            os.path.exists(os.path.join(self.data_dir, 'data')) and \
            os.path.exists(os.path.join(self.data_dir, 'word-vectors')):
            return
        cached_path(self.file_url, self.data_dir)
        files = os.listdir(self.data_dir)
        target_file = ''
        for name in files:
            if name.endswith('.json'):
                target_file = name[:-5]
        try:
            assert target_file in files
        except Exception as e:
            print('allennlp download file error: MDBT Multiwoz data download failed.')
            raise e
        zip_file_path = os.path.join(self.data_dir, target_file+'.zip')
        shutil.copyfile(os.path.join(self.data_dir, target_file), zip_file_path)
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(self.data_dir)


def test_update():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    _config = tf.ConfigProto()
    _config.gpu_options.allow_growth = True
    _config.allow_soft_placement = True
    start_time = time.time()
    mdbt = MultiWozMDBT()
    print('\tMDBT: model build time: {:.2f} seconds'.format(time.time() - start_time))
    mdbt.restore()
    # demo state history
    mdbt.state['history'] = [['null', 'I\'m trying to find an expensive restaurant in the centre part of town.'],
                             ['The Cambridge Chop House is an good expensive restaurant in the centre of town. Would you like me to book it for you?',
                              'Yes, a table for 1 at 16:15 on sunday.  I need the reference number.']]
    new_state = mdbt.update('hi, this is not good')
    print(json.dumps(new_state, indent=4))
    print('all time: {:.2f} seconds'.format(time.time() - start_time))


if __name__ == '__main__':
    loader = AgentDSTDataloader(MultiWOZDataloader())
    data = loader.load_data()
    model = MultiWozMDBT(data=data)
