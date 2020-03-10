# -*- coding: utf-8 -*-
import torch
import os
import json
import zipfile
from convlab2.util.file_util import cached_path
from convlab2.policy.mle.mle import MLEAbstract
from convlab2.policy.rlmodule import MultiDiscretePolicy
from convlab2.policy.vector.vector_crosswoz import CrossWozVector

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DEFAULT_ARCHIVE_FILE = os.path.join(DEFAULT_DIRECTORY, "mle_policy_crosswoz.zip")


class MLE(MLEAbstract):

    def __init__(self,
                 archive_file=DEFAULT_ARCHIVE_FILE,
                 model_file='https://convlab.blob.core.windows.net/convlab-2/mle_policy_crosswoz.zip'):
        root_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json'), 'r') as f:
            cfg = json.load(f)

        voc_file = os.path.join(root_dir, 'data/crosswoz/sys_da_voc.json')
        voc_opp_file = os.path.join(root_dir, 'data/crosswoz/usr_da_voc.json')
        self.vector = CrossWozVector(sys_da_voc_json=voc_file, usr_da_voc_json=voc_opp_file)

        self.policy = MultiDiscretePolicy(self.vector.state_dim, cfg['h_dim'], self.vector.sys_da_dim).to(device=DEVICE)

        if not os.path.isfile(archive_file):
            if not model_file:
                raise Exception("No model for MLE Policy is specified!")
            archive_file = cached_path(model_file)
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'save')
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        if not os.path.exists(os.path.join(model_dir, 'best_mle.pol.mdl')):
            archive = zipfile.ZipFile(archive_file, 'r')
            archive.extractall(model_dir)
        self.load(archive_file, model_file, cfg['load'])
