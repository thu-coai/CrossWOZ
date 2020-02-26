from convlab2.e2e.rnn_rollout.rnnrollout import RNNRolloutAgent
from convlab2.e2e.rnn_rollout.models.rnn_model import RnnModel
from convlab2.e2e.rnn_rollout.models.selection_model import SelectionModel
import convlab2.e2e.rnn_rollout.utils as utils
from convlab2.e2e.rnn_rollout.domain import get_domain
from convlab2 import get_root_path
import os
import zipfile
from convlab2.util.file_util import cached_path
import shutil

class DealornotAgent(RNNRolloutAgent):
    """The Rnn Rollout model for DealorNot dataset."""
    def __init__(self, name, args, sel_args, train=False, diverse=False, max_total_len=100,
                 model_url='https://tatk-data.s3-ap-northeast-1.amazonaws.com/rnnrollout_dealornot.zip'):
        self.config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs')

        self.file_url = model_url

        self.auto_download()

        if not os.path.exists(self.config_path):
            os.mkdir(self.config_path)
        _model_path = os.path.join(self.config_path, 'models')
        self.model_path = _model_path
        if not os.path.exists(_model_path):
            os.makedirs(_model_path)

        self.data_path = os.path.join(get_root_path(), args.data)
        domain = get_domain(args.domain)
        corpus = RnnModel.corpus_ty(domain, self.data_path, freq_cutoff=args.unk_threshold, verbose=True,
                                    sep_sel=args.sep_sel)

        model = RnnModel(corpus.word_dict, corpus.item_dict_old,
                         corpus.context_dict, corpus.count_dict, args)
        state_dict = utils.load_model(os.path.join(self.config_path, args.model_file))  # RnnModel
        model.load_state_dict(state_dict)

        sel_model = SelectionModel(corpus.word_dict, corpus.item_dict_old,
                                   corpus.context_dict, corpus.count_dict, sel_args)
        sel_state_dict = utils.load_model(os.path.join(self.config_path, sel_args.selection_model_file))
        sel_model.load_state_dict(sel_state_dict)

        super(DealornotAgent, self).__init__(model, sel_model, args, name, train, diverse, max_total_len)
        self.vis = args.visual

    def auto_download(self):
        """Automatically download the pretrained model and necessary data."""
        if os.path.exists(os.path.join(self.config_path, 'model/rnn_model_state_dict.th')) and \
            os.path.exists(os.path.join(self.config_path, 'selection_model_state_dict.th')):
            return
        models_dir = os.path.join(self.config_path, 'models')
        cached_path(self.file_url, models_dir)
        files = os.listdir(models_dir)
        target_file = ''
        for name in files:
            if name.endswith('.json'):
                target_file = name[:-5]
        try:
            assert target_file in files
        except Exception as e:
            print('allennlp download file error: RnnRollout Deal_or_Not data download failed.')
            raise e
        zip_file_path = os.path.join(models_dir, target_file + '.zip')
        shutil.copyfile(os.path.join(models_dir, target_file), zip_file_path)
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(models_dir)

def get_context_generator(context_file):
    return utils.ContextGenerator(os.path.join(get_root_path(), context_file))

# if __name__ == '__main__':
#     config_path = './configs'
#     if os.path.exists(os.path.join(config_path, 'model/rnn_model_state_dict.th')) and \
#             os.path.exists(os.path.join(config_path, 'selection_model_state_dict.th')):
#         exit()
#     models_dir = os.path.join(config_path, 'models')
#     file_url = 'https://tatk-data.s3-ap-northeast-1.amazonaws.com/rnnrollout_dealornot.zip'
#     cached_path(file_url, models_dir)
#     files = os.listdir(models_dir)
#     target_file = ''
#     for name in files:
#         if name.endswith('.json'):
#             target_file = name[:-5]
#     try:
#         assert target_file in files
#     except Exception as e:
#         print('allennlp download file error: RnnRollout Deal_or_Not data download failed.')
#         raise e
#     zip_file_path = os.path.join(models_dir, target_file + '.zip')
#     shutil.copyfile(os.path.join(models_dir, target_file), zip_file_path)
#     with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
#         zip_ref.extractall(models_dir)
