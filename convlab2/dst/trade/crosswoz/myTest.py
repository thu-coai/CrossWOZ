import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from .utils.config import *
from .models.TRADE import *
from convlab2.util.file_util import cached_path

'''
python3 myTest.py -ds= -path= -bsz=
'''

# download pretrained model
cached_path(args['model_url'], args['model_path'])
import zipfile
with zipfile.ZipFile(args['model_path'] + 'trade_crosswoz_model.zip', 'r') as zip_ref:
    zip_ref.extractall(args['model_path'])

directory = args['path'].split("/")
HDD = directory[2].split('HDD')[1].split('BSZ')[0]
decoder = directory[1].split('-')[0] 
BSZ = int(args['batch']) if args['batch'] else int(directory[2].split('BSZ')[1].split('DR')[0])
args["decoder"] = decoder
args["HDD"] = HDD
print("HDD", HDD, "decoder", decoder, "BSZ", BSZ)

if args['dataset']=='multiwoz':
    from .utils.utils_multiWOZ_DST import *
else:
    print("You need to provide the --dataset information")

if MODE == 'en':
    train, dev, test, test_special, lang, SLOTS_LIST, gating_dict, max_word = prepare_data_seq(False, args['task'], False, batch_size=BSZ)
else:
    train, dev, test, test_special, lang, SLOTS_LIST, gating_dict, max_word = prepare_data_seq_cn(False, args['task'], False, batch_size=BSZ)

model = globals()[decoder](
    int(HDD), 
    lang=lang, 
    path=args['path'], 
    task=args["task"], 
    lr=0, 
    dropout=0,
    slots=SLOTS_LIST,
    gating_dict=gating_dict,
    nb_train_vocab=max_word)

if args["run_dev_testing"]:
    print("Development Set ...")
    acc_dev = model.evaluate(dev, 1e7, SLOTS_LIST[2]) 

if args['except_domain']!="" and args["run_except_4d"]:
    print("Test Set on 4 domains...")
    acc_test_4d = model.evaluate(test_special, 1e7, SLOTS_LIST[2]) 

print("Test Set ...")
acc_test = model.evaluate(test, 1e7, SLOTS_LIST[3]) 


