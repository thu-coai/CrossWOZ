# specify cuda id
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from convlab2.dst.trade.crosswoz.utils.config import MODE
from tqdm import tqdm
import torch.nn as nn
import shutil, zipfile
from convlab2.util.file_util import cached_path

from convlab2.dst.trade.crosswoz.utils.config import *
from convlab2.dst.trade.crosswoz.models.TRADE import *

'''
python train.py
'''


def download_data(data_url="https://convlab.blob.core.windows.net/convlab-2/trade_crosswoz_data.zip"):
    """Automatically download the pretrained model and necessary data."""
    crosswoz_root = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(os.path.join(crosswoz_root, 'data/crosswoz')) and \
            os.path.exists(os.path.join(crosswoz_root, 'data/dev_dials.json')):
        return
    data_dir = os.path.join(crosswoz_root, 'data')
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    zip_file_path = os.path.join(data_dir, 'trade_crosswoz_data.zip')
    if not os.path.exists(os.path.join(data_dir, 'trade_crosswoz_data.zip')):
        print('downloading crosswoz TRADE data files...')
        cached_path(data_url, data_dir)
        files = os.listdir(data_dir)
        target_file = ''
        for name in files:
            if name.endswith('.json'):
                target_file = name[:-5]
        try:
            assert target_file in files
        except Exception as e:
            print('allennlp download file error: TRADE Cross model download failed.')
            raise e
        shutil.copyfile(os.path.join(data_dir, target_file), zip_file_path)
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        print('unzipping data file ...')
        zip_ref.extractall(data_dir)

early_stop = args['earlyStop']

if args['dataset']=='multiwoz':
    from convlab2.dst.trade.crosswoz.utils.utils_multiWOZ_DST import *
    early_stop = None
else:
    print("You need to provide the --dataset information")
    exit(1)

# specify model parameters
args['decoder'] = 'TRADE'
args['batch'] = 4
args['drop'] = 0.2
args['learn'] = 0.001
args['load_embedding'] = 1

# Configure models and load data
avg_best, cnt, acc = 0.0, 0, 0.0
download_data()
train, dev, test, test_special, lang, SLOTS_LIST, gating_dict, max_word = prepare_data_seq_cn(True, args['task'],
                                                                                              False, batch_size=int(args['batch']))

model = globals()[args['decoder']](
    hidden_size=int(args['hidden']), 
    lang=lang, 
    path=args['path'],
    task=args['task'], 
    lr=float(args['learn']), 
    dropout=float(args['drop']),
    slots=SLOTS_LIST,
    gating_dict=gating_dict, 
    nb_train_vocab=max_word,
    mode=MODE)

# print("[Info] Slots include ", SLOTS_LIST)
# print("[Info] Unpointable Slots include ", gating_dict)

for epoch in range(200):
    print("Epoch:{}".format(epoch))  
    # Run the train function
    pbar = tqdm(enumerate(train),total=len(train))
    for i, data in pbar:
        ## only part data to train
        # if MODE == 'cn' and i >= 1400: break
        model.train_batch(data, int(args['clip']), SLOTS_LIST[1], reset=(i==0))
        model.optimize(args['clip'])
        pbar.set_description(model.print_loss())
        # print(data)
        # exit(1)

    if((epoch+1) % int(args['evalp']) == 0):
        
        acc = model.evaluate(dev, avg_best, SLOTS_LIST[2], early_stop)
        model.scheduler.step(acc)

        if(acc >= avg_best):
            avg_best = acc
            cnt=0
            best_model = model
        else:
            cnt+=1

        if(cnt == args["patience"] or (acc==1.0 and early_stop==None)): 
            print("Ran out of patient, early stop...")  
            break 

