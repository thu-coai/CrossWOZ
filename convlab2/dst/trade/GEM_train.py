from utils.config import *
from models.TRADE import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import quadprog
from copy import deepcopy

## TAKEN FROM https://github.com/facebookresearch/GradientEpisodicMemory/blob/master/model/gem.py
def store_grad(pp, grads, grad_dims, task_id):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
        tid: task id
    """
    # store the gradients
    tid = task_id ### cause babi task are numberd 1 2 .... 20 
    grads[:, tid].fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en, tid].copy_(param.grad.data.view(-1))
        cnt += 1

def overwrite_grad(pp, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    """
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.
        input:  gradient, p-vector
        input:  memories, (t * p)-vector
        output: x, p-vector
    """
    memories_np = memories.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    t = memories_np.shape[0]
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t) + np.eye(t)*0.0000001
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.Tensor(x).view(-1, 1))


#### LOAD MODEL path
except_domain = args['except_domain']
directory = args['path'].split("/")
HDD = directory[2].split('HDD')[1].split('BSZ')[0]
# decoder = directory[1].split('-')[0] 
BSZ = int(args['batch']) if args['batch'] else int(directory[2].split('BSZ')[1].split('DR')[0])
args["decoder"] = "TRADE"
args["HDD"] = HDD

if args['dataset']=='multiwoz':
    from utils.utils_multiWOZ_DST import *
else:
    print("You need to provide the --dataset information")

_, _, test, test_special, lang, SLOTS_LIST, gating_dict, max_word = prepare_data_seq(True, args['task'], False, batch_size=BSZ)

### LOAD DATA
args["data_ratio"] = 1
train_GEM,  _, _, _, _, _, _, _ = prepare_data_seq(True, args['task'], False, batch_size=64)

### finetune on
args['only_domain'] = except_domain
args['except_domain'] = ''
args['fisher_sample'] = 0
train_single, dev_single, test_single, _, _, SLOTS_LIST_single, _, _ = prepare_data_seq(True, args['task'], False, batch_size=BSZ)
args['except_domain'] = except_domain

#### LOAD MODEL 
model = globals()[args["decoder"]](
    int(HDD), 
    lang=lang, 
    path=args['path'], 
    task=args["task"], 
    lr=args["learn"], 
    dropout=args["drop"],
    slots=SLOTS_LIST,
    gating_dict=gating_dict)

print("4 domains test set length used EVAL",len(test_special)*BSZ)
print("4 domains train set length used for GEM",len(train_GEM)*64)
print("1 domains train set length",len(train_single)*BSZ)


n_tasks = 2 ## 1 to store 4 dom and 1 for the final task
grad_dims = []
for param in model.parameters(): grad_dims.append(param.data.numel())
grads = torch.Tensor(sum(grad_dims), n_tasks)
if USE_CUDA: grads = grads.cuda()

avg_best, cnt, acc = 0.0, 0, 0.0
weights_best = deepcopy(model.state_dict())
try:
    for epoch in range(100):
        print("Epoch:{}".format(epoch))  
        # Run the train function
        pbar = tqdm(enumerate(train_single),total=len(train_single))
        for i, data in pbar:

            #### Get Gradient from previous task and store it
            idx_task = 0
            for i, data_o in enumerate(train_GEM):
                model.train_batch(data_o,int(args['clip']), SLOTS_LIST[1], reset=(i==0))
                model.loss_grad.backward()
                store_grad(model.parameters, grads, grad_dims, task_id=idx_task)
                idx_task += 1
                if (i == (n_tasks-2)): break

            model.train_batch(data, int(args['clip']), SLOTS_LIST_single[1], reset=(i==0))
            model.loss_grad.backward()

            store_grad(model.parameters, grads, grad_dims,task_id=n_tasks-1)
            indx = torch.cuda.LongTensor([j for j in range(n_tasks-1) ]) #if USE_CUDA else torch.LongTensor([0])
            dotp = torch.mm(grads[:, n_tasks-1].unsqueeze(0), grads.index_select(1, indx))
            if (dotp < 0).sum() != 0:
                project2cone2(grads[:, n_tasks-1].unsqueeze(1), grads.index_select(1, indx), args['lambda_ewc'])
                # copy gradients back
                overwrite_grad(model.parameters, grads[:, 1], grad_dims)

            model.optimize_GEM(args['clip'])
            
            pbar.set_description(model.print_loss())

        if((epoch+1) % int(args['evalp']) == 0):
            acc = model.evaluate(dev_single, avg_best, SLOTS_LIST_single[2], args["earlyStop"])
            model.scheduler.step(acc)
            if(acc > avg_best):
                avg_best = acc
                cnt=0
                weights_best = deepcopy(model.state_dict())
            else:
                cnt+=1
            if(cnt == 3 or (acc==1.0 and args["earlyStop"]==None)): 
                print("Ran out of patient, early stop...")  
                break 
except KeyboardInterrupt:
    pass

model.load_state_dict({ name: weights_best[name] for name in weights_best })
model.eval()

# After Fine tuning...
print("[Info] After Fine Tune ...")
print("[Info] Test Set on 4 domains...")
acc_test_4d = model.evaluate(test_special, 1e7, SLOTS_LIST[2]) 
print("[Info] Test Set on 1 domain {} ...".format(except_domain))
acc_test = model.evaluate(test_single, 1e7, SLOTS_LIST[3])


