import argparse
import os
import json
import random
import numpy as np
import torch
from convlab2.nlu.jointBERT.dataloader import Dataloader
from convlab2.nlu.jointBERT.jointBERT import JointBERT


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


parser = argparse.ArgumentParser(description="Test a model.")
parser.add_argument('--config_path',
                    help='path to config file')


if __name__ == '__main__':
    args = parser.parse_args()
    config = json.load(open(args.config_path))
    data_dir = config['data_dir']
    output_dir = config['output_dir']
    log_dir = config['log_dir']
    DEVICE = config['DEVICE']

    set_seed(config['seed'])

    if 'multiwoz' in data_dir:
        print('-'*20 + 'dataset:multiwoz' + '-'*20)
        from convlab2.nlu.jointBERT.multiwoz.postprocess import is_slot_da, calculateF1, recover_intent
    elif 'camrest' in data_dir:
        print('-' * 20 + 'dataset:camrest' + '-' * 20)
        from convlab2.nlu.jointBERT.camrest.postprocess import is_slot_da, calculateF1, recover_intent
    elif 'crosswoz' in data_dir:
        print('-' * 20 + 'dataset:crosswoz' + '-' * 20)
        from convlab2.nlu.jointBERT.crosswoz.postprocess import is_slot_da, calculateF1, recover_intent

    intent_vocab = json.load(open(os.path.join(data_dir, 'intent_vocab.json')))
    tag_vocab = json.load(open(os.path.join(data_dir, 'tag_vocab.json')))
    dataloader = Dataloader(intent_vocab=intent_vocab, tag_vocab=tag_vocab,
                            pretrained_weights=config['model']['pretrained_weights'])
    print('intent num:', len(intent_vocab))
    print('tag num:', len(tag_vocab))
    for data_key in ['val', 'test']:
        dataloader.load_data(json.load(open(os.path.join(data_dir, '{}_data.json'.format(data_key)))), data_key,
                             cut_sen_len=0, use_bert_tokenizer=config['use_bert_tokenizer'])
        print('{} set size: {}'.format(data_key, len(dataloader.data[data_key])))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    model = JointBERT(config['model'], DEVICE, dataloader.tag_dim, dataloader.intent_dim)
    model.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model.bin'), DEVICE))
    model.to(DEVICE)
    model.eval()

    batch_size = config['model']['batch_size']

    data_key = 'test'
    predict_golden = {'intent': [], 'slot': [], 'overall': []}
    slot_loss, intent_loss = 0, 0
    for pad_batch, ori_batch, real_batch_size in dataloader.yield_batches(batch_size, data_key=data_key):
        pad_batch = tuple(t.to(DEVICE) for t in pad_batch)
        word_seq_tensor, tag_seq_tensor, intent_tensor, word_mask_tensor, tag_mask_tensor, context_seq_tensor, context_mask_tensor = pad_batch
        if not config['model']['context']:
            context_seq_tensor, context_mask_tensor = None, None

        with torch.no_grad():
            slot_logits, intent_logits, batch_slot_loss, batch_intent_loss = model.forward(word_seq_tensor,
                                                                                           word_mask_tensor,
                                                                                           tag_seq_tensor,
                                                                                           tag_mask_tensor,
                                                                                           intent_tensor,
                                                                                           context_seq_tensor,
                                                                                           context_mask_tensor)
        slot_loss += batch_slot_loss.item() * real_batch_size
        intent_loss += batch_intent_loss.item() * real_batch_size
        for j in range(real_batch_size):
            predicts = recover_intent(dataloader, intent_logits[j], slot_logits[j], tag_mask_tensor[j],
                                      ori_batch[j][0], ori_batch[j][-4])
            labels = ori_batch[j][3]

            predict_golden['overall'].append({
                'predict': predicts,
                'golden': labels
            })
            predict_golden['slot'].append({
                'predict': [x for x in predicts if is_slot_da(x)],
                'golden': [x for x in labels if is_slot_da(x)]
            })
            predict_golden['intent'].append({
                'predict': [x for x in predicts if not is_slot_da(x)],
                'golden': [x for x in labels if not is_slot_da(x)]
            })
        print('[%d|%d] samples' % (len(predict_golden['overall']), len(dataloader.data[data_key])))

    total = len(dataloader.data[data_key])
    slot_loss /= total
    intent_loss /= total
    print('%d samples %s' % (total, data_key))
    print('\t slot loss:', slot_loss)
    print('\t intent loss:', intent_loss)

    for x in ['intent', 'slot', 'overall']:
        precision, recall, F1 = calculateF1(predict_golden[x])
        print('-' * 20 + x + '-' * 20)
        print('\t Precision: %.2f' % (100 * precision))
        print('\t Recall: %.2f' % (100 * recall))
        print('\t F1: %.2f' % (100 * F1))

    output_file = os.path.join(output_dir, 'output.json')
    json.dump(predict_golden['overall'], open(output_file, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
