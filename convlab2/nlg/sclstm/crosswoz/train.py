import argparse
import configparser
import os
import sys
import time
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(root_dir)
import torch
from convlab2.nlg.sclstm.crosswoz.loader.dataset_woz import DatasetWoz, SimpleDatasetWoz
from convlab2.nlg.sclstm.model.lm_deep import LMDeep

USE_CUDA = torch.cuda.is_available()


def score(feat, gen, template):
    """
    feat = ['d-a-s-v:Booking-Book-Day-1', 'd-a-s-v:Booking-Book-Name-1', 'd-a-s-v:Booking-Book-Name-2']
    gen = 'xxx slot-booking-book-name xxx slot-booking-book-time'
    """
    das = []  # e.g. a list of d-a-s-v:Booking-Book-Day
    with open(template) as f:
        for line in f:
            if 'd-a-s-v:' not in line:
                continue
            if '-none' in line or '-?' in line or '-yes' in line or '-no' in line:
                continue
            tok = '-'.join(line.strip().split('-')[:-1])
            if tok not in das:
                das.append(tok)

    total, redunt, miss = 0, 0, 0
    for _das in das:
        feat_count = 0
        das_order = [_das + '-' + str(i) for i in range(20)]
        for _feat in feat:
            if _feat in das_order:
                feat_count += 1
        _das = _das.replace('d-a-s-v:', '').lower().split('-')
        slot_tok = '@' + _das[0][:3] + '-' + _das[1] + '-' + _das[2]

        gen_count = gen.split().count(slot_tok)
        diff_count = gen_count - feat_count
        if diff_count > 0:
            redunt += diff_count
        else:
            miss += -diff_count
        total += feat_count
    return total, redunt, miss


def get_slot_error(dataset, gens, refs, sv_indexes):
    """
    Args:
        gens:  (batch_size, beam_size)
        refs:  (batch_size,)
        sv:    (batch_size,)
    Returns:
        count: accumulative slot error of a batch
        countPerGen: slot error for each sample
    """
    batch_size = len(gens)
    beam_size = len(gens[0])
    assert len(refs) == batch_size and len(sv_indexes) == batch_size

    count = {'total': 0.0, 'redunt': 0.0, 'miss': 0.0}
    countPerGen = [[] for _ in range(batch_size)]
    for batch_idx in range(batch_size):
        for beam_idx in range(beam_size):
            felements = [dataset.cardinality[x + dataset.dfs[2]] for x in sv_indexes[batch_idx]]

            # get slot error per sample(beam)
            total, redunt, miss = score(felements, gens[batch_idx][beam_idx], dataset.template)

            c = {}
            for a, b in zip(['total', 'redunt', 'miss'], [total, redunt, miss]):
                c[a] = b
                count[a] += b
            countPerGen[batch_idx].append(c)

    return count, countPerGen


def evaluate(config, dataset, model, data_type, beam_search, beam_size, batch_size):
    t = time.time()
    model.eval()

    total_loss = 0
    countAll = {'total': 0.0, 'redunt': 0.0, 'miss': 0.0}
    for i in range(dataset.n_batch[data_type]):
        input_var, label_var, feats_var, lengths, refs, featStrs, sv_indexes = dataset.next_batch(data_type=data_type)

        if data_type == 'valid':
            # feed-forward w/i ground truth as input
            decoded_words = model(input_var, dataset, feats_var, gen=False, beam_search=False, beam_size=1)

            # update loss
            loss = model.get_loss(label_var, lengths)
            total_loss += loss.item()

            # run generation for calculating slot error
            decoded_words = model(input_var, dataset, feats_var, gen=True, beam_search=False, beam_size=1)
            countBatch, countPerGen = get_slot_error(dataset, decoded_words, refs, sv_indexes)

        else:  # test
            print('decode batch {} out of {}'.format(i, dataset.n_batch[data_type]), file=sys.stderr)
            decoded_words = model(input_var, dataset, feats_var, gen=True, beam_search=beam_search, beam_size=beam_size)
            countBatch, countPerGen = get_slot_error(dataset, decoded_words, refs, sv_indexes)

            # print generation results
            for batch_idx in range(batch_size):
                print('Feat: {}'.format(featStrs[batch_idx]))
                print('Target: {}'.format(refs[batch_idx]))
                for beam_idx in range(beam_size):
                    c = countPerGen[batch_idx][beam_idx]
                    s = decoded_words[batch_idx][beam_idx]
                    print('Gen{} ({},{},{}): {}'.format(beam_idx, c['redunt'], c['miss'], c['total'], s))
                print('-----------------------------------------------------------')

        # accumulate slot error across batches
        for _type in countAll:
            countAll[_type] += countBatch[_type]

    total_loss /= dataset.n_batch[data_type]

    se = (countAll['redunt'] + countAll['miss']) / countAll['total'] * 100
    print('{} Loss: {:.3f} | Slot error: {:.3f} | Time: {:.1f}'.format(data_type, total_loss, se, time.time() - t))
    print('{} Loss: {:.3f} | Slot error: {:.3f} | Time: {:.1f}'.format(data_type, total_loss, se, time.time() - t),
          file=sys.stderr)
    print('redunt: {}, miss: {}, total: {}'.format(countAll['redunt'], countAll['miss'], countAll['total']))
    print('redunt: {}, miss: {}, total: {}'.format(countAll['redunt'], countAll['miss'], countAll['total']),
          file=sys.stderr)
    return total_loss


def train_epoch(config, dataset, model):
    t = time.time()
    model.train()

    total_loss = 0
    for i in range(dataset.n_batch['train']):
        input_var, label_var, feats_var, lengths, refs, featStrs, sv_indexes = dataset.next_batch()

        # feedforward and calculate loss
        _ = model(input_var, dataset, feats_var)
        loss = model.get_loss(label_var, lengths)

        # update loss
        total_loss += loss.item()

        # update model
        model.update(config.getfloat('MODEL', 'clip'))

    total_loss /= dataset.n_batch['train']

    print('Train Loss: {:.3f} | Time: {:.1f}'.format(total_loss, time.time() - t))
    print('Train Loss: {:.3f} | Time: {:.1f}'.format(total_loss, time.time() - t), file=sys.stderr)


def read(config, args, mode):
    # get data
    print('Processing data...', file=sys.stderr)

    # TODO: remove this constraint
    if mode == 'test' and args.beam_search:
        print('Set batch_size to 1 due to beam search')
        print('Set batch_size to 1 due to beam search', file=sys.stderr)

    percentage = args.percent
    dataset = DatasetWoz(config, percentage=percentage, use_cuda=USE_CUDA)

    # get model hyper-parameters
    n_layer = args.n_layer
    hidden_size = config.getint('MODEL', 'hidden_size')
    dropout = config.getfloat('MODEL', 'dropout')
    lr = args.lr
    beam_size = args.beam_size

    # get feat size
    d_size = dataset.do_size + dataset.da_size + dataset.sv_size  # len of 1-hot feat
    vocab_size = len(dataset.word2index)

    model_path = args.model_path
    model = LMDeep('sclstm', vocab_size, vocab_size, hidden_size, d_size, n_layer=n_layer, dropout=dropout, lr=lr,
                   use_cuda=USE_CUDA)

    if mode == 'train':
        print('do not support re-train model, please make sure the model do not exist before training')
        assert not os.path.isfile(model_path)

    else:  # mode = 'test'
        # load model
        print(model_path, file=sys.stderr)
        assert os.path.isfile(model_path)
        model.load_state_dict(torch.load(model_path))
        if mode != 'adapt':
            model.eval()

    # Print model info
    print('\n***** MODEL INFO *****')
    print('MODEL PATH:', model_path)
    print('SIZE OF HIDDEN:', hidden_size)
    print('# of LAYER:', n_layer)
    print('SAMPLE/BEAM SIZE:', beam_size)
    print('*************************\n')

    # Move models to GPU
    if USE_CUDA:
        model.cuda()

    return dataset, model


def train(config, args):
    dataset, model = read(config, args, args.mode)
    n_layer = args.n_layer
    model_path = args.model_path

    # Start training
    epoch = 0
    best_loss = 10000
    print('Start training', file=sys.stderr)
    while epoch < config.getint('TRAINING', 'n_epochs'):
        dataset.reset()
        print('Epoch', epoch, '(n_layer {})'.format(n_layer))
        print('Epoch', epoch, '(n_layer {})'.format(n_layer), file=sys.stderr)
        train_epoch(config, dataset, model)

        loss = evaluate(config, dataset, model, 'valid', False, 1, dataset.batch_size)

        if loss < best_loss:
            earlyStop = 0
            # save model
            print('Best loss: {:.3f}, AND Save model!'.format(loss))
            print('Best loss: {:.3f}, AND Save model!'.format(loss), file=sys.stderr)
            torch.save(model.state_dict(), model_path)
            best_loss = loss
        else:
            earlyStop += 1

        if earlyStop == 6:  # do not improve on dev set 10 epoches in a row
            return
        epoch += 1
        print('----------------------------------------')
        print('----------------------------------------', file=sys.stderr)


def test(config, args):
    dataset, model = read(config, args, 'test')
    data_type = args.mode
    beam_search = args.beam_search
    beam_size = args.beam_size

    print('TEST ON: {}'.format(data_type))
    print('TEST ON: {}'.format(data_type), file=sys.stderr)

    # Evaluate model
    loss = evaluate(config, dataset, model, 'test', beam_search, beam_size, dataset.batch_size)


def interact(config, args):
    dataset = SimpleDatasetWoz(config)

    # get model hyper-parameters
    n_layer = args.n_layer
    hidden_size = config.getint('MODEL', 'hidden_size')
    dropout = config.getfloat('MODEL', 'dropout')
    lr = args.lr
    beam_size = args.beam_size

    # get feat size
    d_size = dataset.do_size + dataset.da_size + dataset.sv_size  # len of 1-hot feat
    vocab_size = len(dataset.word2index)

    model = LMDeep('sclstm', vocab_size, vocab_size, hidden_size, d_size, n_layer=n_layer, dropout=dropout, lr=lr)
    model_path = args.model_path
    print(model_path)
    assert os.path.isfile(model_path)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Move models to GPU
    if USE_CUDA:
        model.cuda()

    data_type = args.mode
    print('INTERACT ON: {}'.format(data_type))

    example = {"Attraction-Inform": [["Choice", "many"], ["Area", "centre of town"]],
               "Attraction-Select": [["Type", "church"], ["Type", " swimming"], ["Type", " park"]]}
    for k, v in example.items():
        counter = {}
        for pair in v:
            if pair[0] in counter:
                counter[pair[0]] += 1
            else:
                counter[pair[0]] = 1
            pair.insert(1, str(counter[pair[0]]))

    do_idx, da_idx, sv_idx, featStr = dataset.getFeatIdx(example)
    do_cond = [1 if i in do_idx else 0 for i in range(dataset.do_size)]  # domain condition
    da_cond = [1 if i in da_idx else 0 for i in range(dataset.da_size)]  # dial act condition
    sv_cond = [1 if i in sv_idx else 0 for i in range(dataset.sv_size)]  # slot/value condition
    feats = [do_cond + da_cond + sv_cond]

    feats_var = torch.FloatTensor(feats)
    if USE_CUDA:
        feats_var = feats_var.cuda()

    decoded_words = model.generate(dataset, feats_var, beam_size)
    delex = decoded_words[0]  # (beam_size)

    recover = []
    for sen in delex:
        counter = {}
        words = sen.split()
        for word in words:
            if word.startswith('slot-'):
                flag = True
                _, domain, intent, slot_type = word.split('-')
                da = domain.capitalize() + '-' + intent.capitalize()
                if da in example:
                    key = da + '-' + slot_type.capitalize()
                    for pair in example[da]:
                        if (pair[0].lower() == slot_type) and (
                                (key not in counter) or (counter[key] == int(pair[1]) - 1)):
                            sen = sen.replace(word, pair[2], 1)
                            counter[key] = int(pair[1])
                            flag = False
                            break
                if flag:
                    sen = sen.replace(word, '', 1)
        recover.append(sen)

    print('meta', example)
    for i in range(beam_size):
        print(i, delex[i])
        print(i, recover[i])


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse():
    parser = argparse.ArgumentParser(description='Train dialogue generator')
    parser.add_argument('--mode', type=str, default='interact', help='train or test')
    parser.add_argument('--model_path', type=str, default='sclstm.pt', help='saved model path')
    parser.add_argument('--n_layer', type=int, default=1, help='# of layers in LSTM')
    parser.add_argument('--percent', type=float, default=1, help='percentage of training data')
    parser.add_argument('--beam_search', type=str2bool, default=False, help='beam_search')
    parser.add_argument('--attn', type=str2bool, default=True, help='whether to use attention or not')
    parser.add_argument('--beam_size', type=int, default=10, help='number of generated sentences')
    parser.add_argument('--bs', type=int, default=256, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0025, help='learning rate')
    parser.add_argument('--user', type=str2bool, default=False, help='use user data')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    if args.user:
        config.read('config/config_usr.cfg')
    else:
        config.read('config/config.cfg')
    config.set('DATA', 'dir', os.path.dirname(os.path.abspath(__file__)))

    return args, config


if __name__ == '__main__':
    # # set seed for reproducing
    # random.seed(1235)
    # torch.manual_seed(1235)
    # torch.cuda.manual_seed_all(1235)

    args, config = parse()

    # training
    if args.mode == 'train' or args.mode == 'adapt':
        train(config, args)
    # test
    elif args.mode == 'test':
        test(config, args)
    # interact
    elif args.mode == 'interact':
        interact(config, args)
    else:
        print('mode cannot be recognized')
        sys.exit(1)
