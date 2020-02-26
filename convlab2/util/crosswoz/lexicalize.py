def delexicalize_da(da):
    delexicalized_da = []
    counter = {}
    for intent, domain, slot, value in da:
        if intent in ['Inform', 'Recommend']:
            key = '+'.join([intent, domain, slot])
            counter.setdefault(key,0)
            counter[key] += 1
            delexicalized_da.append(key+'+'+str(counter[key]))
        else:
            delexicalized_da.append('+'.join([intent, domain, slot, value]))
    
    return delexicalized_da


def lexicalize_da(da, cur_domain, entities):
    not_dish = {'当地口味', '老字号', '其他', '美食林风味', '特色小吃', '美食林臻选', '深夜营业', '名人光顾', '四合院'}
    lexicalized_da = []
    for a in da:
        intent, domain, slot, value = a.split('+')
        if intent in ['General', 'NoOffer']:
            lexicalized_da.append([intent, domain, slot, value])
        elif domain==cur_domain:
            value = int(value)-1
            if domain == '出租':
                assert intent=='Inform'
                assert slot in ['车型', '车牌']
                assert value == 0
                value = entities[0][1][slot]
                lexicalized_da.append([intent, domain, slot, value])
            elif domain == '地铁':
                assert intent=='Inform'
                assert slot in ['出发地附近地铁站', '目的地附近地铁站']
                assert value == 0
                if slot == '出发地附近地铁站':
                    candidates = [v for n, v in entities if '起点' in n]
                    if candidates:
                        value = candidates[0]['地铁']
                    else:
                        value = '无'
                else:
                    candidates = [v for n, v in entities if '终点' in n]
                    if candidates:
                        value = candidates[0]['地铁']
                    else:
                        value = '无'
                lexicalized_da.append([intent, domain, slot, value])
            else:
                if intent=='Recommend':
                    assert slot=='名称'
                    if len(entities)>value:
                        value = entities[value][0]
                        lexicalized_da.append([intent, domain, slot, value])
                else:
                    assert intent=='Inform'
                    if len(entities)>value:
                        entity = entities[0][1]
                        if '周边' in slot:
                            assert isinstance(entity[slot], list)
                            if value < len(entity[slot]):
                                value = entity[slot][value]
                                lexicalized_da.append([intent, domain, slot, value])
                        elif slot=='推荐菜':
                            assert isinstance(entity[slot], list)
                            dishes = [x for x in entity[slot] if x not in not_dish]
                            if len(dishes)>value:
                                value = dishes[value]
                            lexicalized_da.append([intent, domain, slot, value])
                        elif '酒店设施' in slot:
                            assert value == 0
                            slot, value = slot.split('-')
                            assert isinstance(entity[slot], list)
                            if value in entity[slot]:
                                lexicalized_da.append([intent, domain, '-'.join([slot, value]), '是'])
                            else:
                                lexicalized_da.append([intent, domain, '-'.join([slot, value]), '否'])
                        elif slot in ['门票', '价格', '人均消费']:
                            assert value == 0
                            value = entity[slot]
                            lexicalized_da.append([intent, domain, slot, '{}元'.format(value)])
                        elif slot == '评分':
                            assert value == 0
                            value = entity[slot]
                            lexicalized_da.append([intent, domain, slot, '{}分'.format(value)])
                        else:
                            assert value == 0
                            value = entity[slot]
                            lexicalized_da.append([intent, domain, slot, value])
    return lexicalized_da
