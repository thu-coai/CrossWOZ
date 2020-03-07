import re
import torch


def is_slot_da(da):
    tag_da = {'Inform', 'Recommend'}
    not_tag_slot = '酒店设施'
    if da[0] in tag_da and not_tag_slot not in da[2]:
        return True
    return False


def calculateF1(predict_golden):
    TP, FP, FN = 0, 0, 0
    for item in predict_golden:
        predicts = item['predict']
        labels = item['golden']
        for ele in predicts:
            if ele in labels:
                TP += 1
            else:
                FP += 1
        for ele in labels:
            if ele not in predicts:
                FN += 1
    # print(TP, FP, FN)
    precision = 1.0 * TP / (TP + FP) if TP + FP else 0.
    recall = 1.0 * TP / (TP + FN) if TP + FN else 0.
    F1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.
    return precision, recall, F1


def tag2das(word_seq, tag_seq):
    assert len(word_seq)==len(tag_seq)
    das = []
    i = 0
    while i < len(tag_seq):
        tag = tag_seq[i]
        if tag.startswith('B'):
            intent, domain, slot = tag[2:].split('+')
            value = word_seq[i]
            j = i + 1
            while j < len(tag_seq):
                if tag_seq[j].startswith('I') and tag_seq[j][2:] == tag[2:]:
                    # tag_seq[j][2:].split('+')[-1]==slot or tag_seq[j][2:] == tag[2:]
                    if word_seq[j].startswith('##'):
                        value += word_seq[j][2:]
                    else:
                        value += word_seq[j]
                    i += 1
                    j += 1
                else:
                    break
            das.append([intent, domain, slot, value])
        i += 1
    return das


def intent2das(intent_seq):
    triples = []
    for intent in intent_seq:
        intent, domain, slot, value = re.split('\+', intent)
        triples.append([intent, domain, slot, value])
    return triples


def recover_intent(dataloader, intent_logits, tag_logits, tag_mask_tensor, ori_word_seq, new2ori):
    # tag_logits = [sequence_length, tag_dim]
    # intent_logits = [intent_dim]
    # tag_mask_tensor = [sequence_length]
    max_seq_len = tag_logits.size(0)
    das = []
    for j in range(dataloader.intent_dim):
        if intent_logits[j] > 0:
            intent, domain, slot, value = re.split('\+', dataloader.id2intent[j])
            das.append([intent, domain, slot, value])
    tags = []
    for j in range(1 , max_seq_len -1):
        if tag_mask_tensor[j] == 1:
            value, tag_id = torch.max(tag_logits[j], dim=-1)
            tags.append(dataloader.id2tag[tag_id.item()])
    tag_intent = tag2das(ori_word_seq, tags)
    das += tag_intent
    return das
