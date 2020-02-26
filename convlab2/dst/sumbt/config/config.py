import os


DATASET = 'multiwoz'

DATA_DIR = os.path.join(os.path.split(os.path.realpath(__file__))[0], '../../../../data/multiwoz/')
TMP_DATA_DIR = os.path.join(os.path.split(os.path.realpath(__file__))[0], '../tmp_data/')
SELF_DATA_DIR = os.path.join(os.path.split(os.path.realpath(__file__))[0], '../data/')
BERT_MODEL = 'bert-base-uncased'
BERT_DIR = '/home/libing/pytorch-pretrained-bert/bert-base-uncased'  # todo: how?
# BERT_DIR = ''
TASK_NAME = 'bert-gru-sumbt'
OUTPUT_DIR = os.path.join(os.path.split(os.path.realpath(__file__))[0], '../multiwoz/output/')
TARGET_SLOT = 'all'
DO_LOWER_CASE = True

HIDDEN_DIM = 300
RNN_NUM_LAYERS = 1
ZERO_INIT_RNN = False
MAX_SEQ_LENGTH = 64
MAX_LABEL_LENGTH = 32
MAX_TURN_LENGTH = 22
ATTN_HEAD = 4
DEVICE = 'cuda'
N_GPU = 2
DISTANCE_METRIC = 'euclidean'
BATCH_SIZE = 4
LEARNING_RATE = 5e-5
EPOCHS = 300
PATIENCE = 10.0
WARM_UP_PROPORTION = 0.1
SEED = 42
FP16 = False
FP16_LOSS_SCALE = 0
FIX_UTTERANCE_ENCODER = False
GRADIENT_ACCUM_STEPS = 1
TENSORBOARD = True
