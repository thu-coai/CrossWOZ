from copy import deepcopy


def delexicalize_da(da, requestable):
    delexicalized_da = []
    counter = {}
    for intent, slot, value in da:
        if intent in requestable:
            v = '?'
        else:
            if slot == 'none':
                v = 'none'
            else:
                k = '-'.join([intent, slot])
                counter.setdefault(k, 0)
                counter[k] += 1
                v = str(counter[k])
        delexicalized_da.append([intent, slot, v])
    return delexicalized_da


def flat_da(delexicalized_da):
    flaten = ['-'.join(x) for x in delexicalized_da]
    return flaten


def deflat_da(meta):
    meta = deepcopy(meta)
    dialog_act = {}
    for da in meta:
        i, s, v = da.split('-')
        k = i
        if k not in dialog_act:
            dialog_act[k] = []
        dialog_act[k].append([s, v])
    return dialog_act


def lexicalize_da(meta, entities, state, requestable):
    meta = deepcopy(meta)

    for k, v in meta.items():
        intent = k
        if intent in requestable:
            for pair in v:
                pair[1] = '?'
        elif intent.lower() in ['nooffer', 'nobook']:
            for pair in v:
                if pair[0] in state:
                    pair[1] = state[pair[0]]
                else:
                    pair[1] = 'none'
        else:
            for pair in v:
                if pair[1] == 'none':
                    continue
                elif pair[0].lower() == 'choice':
                    pair[1] = str(len(entities))
                else:
                    n = int(pair[1]) - 1
                    if len(entities) > n and pair[0] in entities[n]:
                        pair[1] = entities[n][pair[0]]
                    else:
                        pair[1] = 'none'
    tuples = []
    for intent, svs in meta.items():
        for slot, value in svs:
            tuples.append([intent, slot, value])
    return tuples
