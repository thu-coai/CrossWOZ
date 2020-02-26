import logging
import os
import time
import json


class _Config:
    def __init__(self):
        self._init_logging_handler()
        self.eos_m_token = 'EOS_M'       
        self.beam_len_bonus = 0.5

        self.mode = 'unknown'
        self.m = 'TSD'
        self.prev_z_method = 'none'

        self.seed = 0
  
    def init_handler(self, tsdf_init_config):
        self.__dict__.update(tsdf_init_config)
        self.vocab_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.vocab_path)
        self.entity = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.entity)
        self.glove_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.glove_path)
        self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.model_path)
        self.result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.result_path)

        if tsdf_init_config['dataset']=='multiwoz':
            self.train = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.train)
            self.dev = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.dev)
            self.test = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.test)
            for i, db in enumerate(self.db):
                self.db[i] = os.path.join(os.path.dirname(os.path.abspath(__file__)), db)
        elif tsdf_init_config['dataset']=='camrest':
            self.data = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.data)
            self.db = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.db)

        # init_method = {
        #     'tsdf-camrest':self._camrest_tsdf_init,
        #     'tsdf-kvret':self._kvret_tsdf_init,
        #     'tsdf-multiwoz':self._multiwoz_tsdf_init
        # }
        # init_method[m]()

    # def _camrest_tsdf_init(self):
    #     self.beam_len_bonus = 0.5
    #     self.prev_z_method = 'separate'
    #     self.vocab_size = 800 #840
    #     self.embedding_size = 50
    #     self.hidden_size = 50
    #     self.split = (3, 1, 1)
    #     self.lr = 0.003
    #     self.lr_decay = 0.5
    #     self.vocab_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vocab/vocab-camrest.pkl')
    #     self.data = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/CamRest676/CamRest676.json')
    #     self.entity = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/CamRest676/CamRestOTGY.json')
    #     self.db = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/CamRest676/CamRestDB.json')
    #     self.glove_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/glove/glove.6B.50d.txt')
    #     self.batch_size = 32
    #     self.z_length = 8
    #     self.degree_size = 5
    #     self.layer_num = 1
    #     self.dropout_rate = 0.5
    #     self.epoch_num = 100 # triggered by early stop
    #     self.rl_epoch_num = 2
    #     self.cuda = False
    #     self.spv_proportion = 100
    #     self.max_ts = 40
    #     self.early_stop_count = 3
    #     self.new_vocab = True
    #     self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/camrest.pkl')
    #     self.result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results/camrest-rl.csv')
    #     self.teacher_force = 100
    #     self.beam_search = False
    #     self.beam_size = 10
    #     self.sampling = False
    #     self.unfrz_attn_epoch = 0
    #     self.skip_unsup = False
    #     self.truncated = False
    #     self.pretrain = False
    #
    # def _kvret_tsdf_init(self):
    #     self.prev_z_method = 'separate'
    #     self.intent = 'all'
    #     self.vocab_size = 1400
    #     self.embedding_size = 50
    #     self.hidden_size = 50
    #     self.split = None
    #     self.lr = 0.003
    #     self.lr_decay = 0.5
    #     self.vocab_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vocab/vocab-kvret.pkl')
    #     self.train = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/kvret/kvret_train_public.json')
    #     self.dev = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/kvret/kvret_dev_public.json')
    #     self.test = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/kvret/kvret_test_public.json')
    #     self.entity = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/kvret/kvret_entities.json')
    #     self.glove_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/glove/glove.6B.50d.txt')
    #     self.batch_size = 32
    #     self.degree_size = 5
    #     self.z_length = 8
    #     self.layer_num = 1
    #     self.dropout_rate = 0.5
    #     self.epoch_num = 100
    #     self.cuda = False
    #     self.spv_proportion = 100
    #     self.alpha = 0.0
    #     self.max_ts = 40
    #     self.early_stop_count = 3
    #     self.new_vocab = True
    #     self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/kvret.pkl')
    #     self.result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results/kvret.csv')
    #     self.teacher_force = 100
    #     self.beam_search = False
    #     self.beam_size = 10
    #     self.sampling = False
    #     self.unfrz_attn_epoch = 0
    #     self.skip_unsup = False
    #     self.truncated = False
    #     self.pretrain = False
    #     self.oov_proportion = 100
    #
    # def _multiwoz_tsdf_init(self):
    #     self.beam_len_bonus = 0.5
    #     self.prev_z_method = 'separate'
    #     self.vocab_size = 4000 #9380
    #     self.embedding_size = 50
    #     self.hidden_size = 50
    #     self.split = (3, 1, 1)
    #     self.lr = 0.003
    #     self.lr_decay = 0.5
    #     self.vocab_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vocab/vocab-multiwoz.pkl')
    #     self.train = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/MultiWoz/train.json')
    #     self.dev = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/MultiWoz/valid.json')
    #     self.test = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/MultiWoz/test.json')
    #     self.entity = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/MultiWoz/entities.json')
    #     self.db = [os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/MultiWoz/attraction_db.json'),
    #                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/MultiWoz/hotel_db.json'),
    #                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/MultiWoz/restaurant_db.json'),
    #                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/MultiWoz/hospital_db.json'),
    #                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/MultiWoz/train_db.json')]
    #     self.glove_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/glove/glove.6B.50d.txt')
    #     self.batch_size = 32
    #     self.z_length = 8
    #     self.degree_size = 5
    #     self.layer_num = 1
    #     self.dropout_rate = 0.5
    #     self.epoch_num = 100 # triggered by early stop
    #     self.rl_epoch_num = 2
    #     self.cuda = True
    #     self.spv_proportion = 100
    #     self.max_ts = 40
    #     self.early_stop_count = 3
    #     self.new_vocab = True
    #     self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/multiwoz.pkl')
    #     self.result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results/multiwoz.csv')
    #     self.teacher_force = 100
    #     self.beam_search = False
    #     self.beam_size = 10
    #     self.sampling = False
    #     self.unfrz_attn_epoch = 0
    #     self.skip_unsup = False
    #     self.truncated = False
    #     self.pretrain = False

    def __str__(self):
        s = ''
        for k,v in self.__dict__.items():
            s += '{} : {}\n'.format(k,v)
        return s

    def _init_logging_handler(self):
        current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

        stderr_handler = logging.StreamHandler()
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'log')
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        file_handler = logging.FileHandler(os.path.join(log_dir, 'log_{}.txt').format(current_time))
        logging.basicConfig(handlers=[stderr_handler, file_handler])
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

global_config = _Config()

