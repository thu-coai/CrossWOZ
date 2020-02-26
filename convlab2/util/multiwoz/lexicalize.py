from copy import deepcopy
from convlab2.util.multiwoz.multiwoz_slot_trans import REF_SYS_DA


def delexicalize_da(da, requestable):
    delexicalized_da = []
    counter = {}
    for intent, domain, slot, value in da:
        if intent in requestable:
            v = '?'
        else:
            if slot=='none':
                v = 'none'
            else:
                k = '-'.join([intent, domain, slot])
                counter.setdefault(k, 0)
                counter[k] += 1
                v = str(counter[k])
        delexicalized_da.append([domain, intent, slot, v])
    return delexicalized_da


def flat_da(delexicalized_da):
    flaten = ['-'.join(x) for x in delexicalized_da]
    return flaten


def deflat_da(meta):
    meta = deepcopy(meta)
    dialog_act = {}
    for da in meta:
        d, i, s, v = da.split('-')
        k = '-'.join((d, i))
        if k not in dialog_act:
            dialog_act[k] = []
        dialog_act[k].append([s, v])
    return dialog_act


def lexicalize_da(meta, entities, state, requestable):
    meta = deepcopy(meta)

    for k, v in meta.items():
        domain, intent = k.split('-')
        if domain.lower() in ['general', 'booking']:
            continue
        elif intent in requestable:
            for pair in v:
                pair[1] = '?'
        elif intent.lower() in ['nooffer', 'nobook']:
            for pair in v:
                if pair[0] in state[domain.lower()]['semi']:
                    pair[1] = state[domain.lower()]['semi'][pair[0]]
                else:
                    pair[1] = 'none'
        else:
            for pair in v:
                if pair[1] == 'none':
                    continue
                elif pair[0].lower() == 'choice':
                    pair[1] = str(len(entities[domain]))
                else:
                    n = int(pair[1]) - 1
                    if len(entities[domain]) > n and pair[0] in REF_SYS_DA[domain] and REF_SYS_DA[domain][pair[0]] in \
                            entities[domain][n]:
                        pair[1] = entities[domain][n][REF_SYS_DA[domain][pair[0]]]
                    else:
                        pair[1] = 'none'
    tuples = []
    for domain_intent, svs in meta.items():
        for slot, value in svs:
            domain, intent = domain_intent.split('-')
            tuples.append([intent, domain, slot, value])
    return tuples
