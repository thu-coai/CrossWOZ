import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from convlab2.dst.trade.crosswoz.utils.config import MODE
from tqdm import tqdm
import torch.nn as nn

from convlab2.dst.trade.crosswoz.utils.config import *
from convlab2.dst.trade.crosswoz.models.TRADE import *

'''
python myTrain.py -dec= -bsz= -hdd= -dr= -lr=
'''


early_stop = args['earlyStop']

if args['dataset']=='multiwoz':
    from convlab2.dst.trade.crosswoz.utils.utils_multiWOZ_DST import *
    early_stop = None
else:
    print("You need to provide the --dataset information")
    exit(1)

# Configure models and load data
avg_best, cnt, acc = 0.0, 0, 0.0
if MODE == 'en':
    train, dev, test, test_special, lang, SLOTS_LIST, gating_dict, max_word = prepare_data_seq(True, args['task'], False, batch_size=int(args['batch']))
else:
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

