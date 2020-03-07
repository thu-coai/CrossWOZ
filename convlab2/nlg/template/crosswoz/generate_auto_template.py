import json
from collections import defaultdict
from copy import copy
import functools
import zipfile


def read_zipped_json(filepath, filename):
    archive = zipfile.ZipFile(filepath, 'r')
    return json.load(archive.open(filename))


file_path = '../../../../data/crosswoz/train.json.zip'
data = read_zipped_json(file_path, 'train.json')

print('\n\nLength of data: ', len(data))

user_multi_intent_dict = defaultdict(list)
sys_multi_intent_dict = defaultdict(list)

role = None


def cmp_intent(intent1: str, intent2: str):
    assert role in ['sys', 'usr']
    intent_order = {
        'usr': (
            'General+greet+none',
            'Inform+出租+出发地',
            'Inform+出租+目的地',
            'Inform+地铁+出发地',
            'Inform+地铁+目的地',
            'Inform+景点+名称',
            'Inform+景点+游玩时间',
            'Inform+景点+评分',
            'Inform+景点+门票',
            'Inform+景点+门票+免费',
            'Inform+酒店+价格',
            'Inform+酒店+名称',
            'Inform+酒店+评分',
            'Inform+酒店+酒店类型',
            'Inform+酒店+酒店设施+否',
            'Inform+酒店+酒店设施+是',
            'Inform+餐馆+人均消费',
            'Inform+餐馆+名称',
            'Inform+餐馆+推荐菜',
            'Inform+餐馆+推荐菜1+推荐菜2',
            'Inform+餐馆+评分',
            'Select+景点+源领域+景点',
            'Select+景点+源领域+酒店',
            'Select+景点+源领域+餐馆',
            'Select+酒店+源领域+景点',
            'Select+酒店+源领域+餐馆',
            'Select+餐馆+源领域+景点',
            'Select+餐馆+源领域+酒店',
            'Select+餐馆+源领域+餐馆',
            'Request+出租+车型',
            'Request+出租+车牌',
            'Request+地铁+出发地附近地铁站',
            'Request+地铁+目的地附近地铁站',
            'Request+景点+名称',
            'Request+景点+周边景点',
            'Request+景点+周边酒店',
            'Request+景点+周边餐馆',
            'Request+景点+地址',
            'Request+景点+游玩时间',
            'Request+景点+电话',
            'Request+景点+评分',
            'Request+景点+门票',
            'Request+酒店+价格',
            'Request+酒店+名称',
            'Request+酒店+周边景点',
            'Request+酒店+周边餐馆',
            'Request+酒店+地址',
            'Request+酒店+电话',
            'Request+酒店+评分',
            'Request+酒店+酒店类型',
            'Request+酒店+酒店设施',
            'Request+餐馆+人均消费',
            'Request+餐馆+名称',
            'Request+餐馆+周边景点',
            'Request+餐馆+周边酒店',
            'Request+餐馆+周边餐馆',
            'Request+餐馆+地址',
            'Request+餐馆+推荐菜',
            'Request+餐馆+电话',
            'Request+餐馆+营业时间',
            'Request+餐馆+评分',
            'General+thank+none',
            'General+bye+none'
        ),
        'sys': (
            'General+greet+none',
            'General+thank+none',
            'General+welcome+none',
            'NoOffer+景点+none',
            'NoOffer+酒店+none',
            'NoOffer+餐馆+none',
            'Inform+主体+属性+无',
            'Inform+出租+车型',
            'Inform+出租+车牌',
            'Inform+地铁+出发地附近地铁站',
            'Inform+地铁+目的地附近地铁站',
            'Inform+景点+名称',
            'Inform+景点+周边景点',
            'Inform+景点+周边景点1+周边景点2',
            'Inform+景点+周边景点1+周边景点2+周边景点3',
            'Inform+景点+周边景点1+周边景点2+周边景点3+周边景点4',
            'Inform+景点+周边酒店',
            'Inform+景点+周边酒店1+周边酒店2',
            'Inform+景点+周边酒店1+周边酒店2+周边酒店3',
            'Inform+景点+周边酒店1+周边酒店2+周边酒店3+周边酒店4',
            'Inform+景点+周边餐馆',
            'Inform+景点+周边餐馆1+周边餐馆2',
            'Inform+景点+周边餐馆1+周边餐馆2+周边餐馆3',
            'Inform+景点+周边餐馆1+周边餐馆2+周边餐馆3+周边餐馆4',
            'Inform+景点+地址',
            'Inform+景点+游玩时间',
            'Inform+景点+电话',
            'Inform+景点+评分',
            'Inform+景点+门票',
            'Inform+景点+门票+免费',
            'Inform+酒店+价格',
            'Inform+酒店+名称',
            'Inform+酒店+周边景点',
            'Inform+酒店+周边景点1+周边景点2',
            'Inform+酒店+周边景点1+周边景点2+周边景点3',
            'Inform+酒店+周边景点1+周边景点2+周边景点3+周边景点4',
            'Inform+酒店+周边餐馆',
            'Inform+酒店+周边餐馆1+周边餐馆2',
            'Inform+酒店+周边餐馆1+周边餐馆2+周边餐馆3',
            'Inform+酒店+周边餐馆1+周边餐馆2+周边餐馆3+周边餐馆4',
            'Inform+酒店+地址',
            'Inform+酒店+电话',
            'Inform+酒店+评分',
            'Inform+酒店+酒店类型',
            'Inform+酒店+酒店设施+否',
            'Inform+酒店+酒店设施+是',
            'Inform+餐馆+人均消费',
            'Inform+餐馆+名称',
            'Inform+餐馆+周边景点',
            'Inform+餐馆+周边景点1+周边景点2',
            'Inform+餐馆+周边景点1+周边景点2+周边景点3',
            'Inform+餐馆+周边景点1+周边景点2+周边景点3+周边景点4',
            'Inform+餐馆+周边酒店',
            'Inform+餐馆+周边酒店1+周边酒店2',
            'Inform+餐馆+周边酒店1+周边酒店2+周边酒店3',
            'Inform+餐馆+周边酒店1+周边酒店2+周边酒店3+周边酒店4',
            'Inform+餐馆+周边餐馆',
            'Inform+餐馆+周边餐馆1+周边餐馆2',
            'Inform+餐馆+周边餐馆1+周边餐馆2+周边餐馆3',
            'Inform+餐馆+周边餐馆1+周边餐馆2+周边餐馆3+周边餐馆4',
            'Inform+餐馆+地址',
            'Inform+餐馆+推荐菜',
            'Inform+餐馆+推荐菜1+推荐菜2',
            'Inform+餐馆+推荐菜1+推荐菜2+推荐菜3',
            'Inform+餐馆+推荐菜1+推荐菜2+推荐菜3+推荐菜4',
            'Inform+餐馆+电话',
            'Inform+餐馆+营业时间',
            'Inform+餐馆+评分',
            'Recommend+景点+名称',
            'Recommend+景点+名称1+名称2',
            'Recommend+景点+名称1+名称2+名称3',
            'Recommend+景点+名称1+名称2+名称3+名称4',
            'Recommend+酒店+名称',
            'Recommend+酒店+名称1+名称2',
            'Recommend+酒店+名称1+名称2+名称3',
            'Recommend+酒店+名称1+名称2+名称3+名称4',
            'Recommend+餐馆+名称',
            'Recommend+餐馆+名称1+名称2',
            'Recommend+餐馆+名称1+名称2+名称3',
            'Recommend+餐馆+名称1+名称2+名称3+名称4',
            'General+reqmore+none',
            'General+bye+none'
        )
    }
    intent1 = intent1.split('1')[0]
    intent2 = intent2.split('1')[0]
    if 'Inform' in intent1 and '无' in intent1:
        intent1 = 'Inform+主体+属性+无'
    if 'Inform' in intent2 and '无' in intent2:
        intent2 = 'Inform+主体+属性+无'
    try:
        assert intent1 in intent_order[role] and intent2 in intent_order[role]
    except AssertionError:
        print(role, intent1, intent2)
    return intent_order[role].index(intent1) - intent_order[role].index(intent2)

dialogue_id = 1
for dialogue in data.values():
    print('Processing the %dth dialogue' % dialogue_id)
    dialogue_id += 1
    for round in dialogue['messages']:
        # original content
        content = round['content']
        intent_list = []
        intent_frequency = defaultdict(int)
        role = round['role']
        usable = True
        for act in round['dialog_act']:
            cur_act = copy(act)

            # intent list
            facility = None  # for 酒店设施
            if '酒店设施' in cur_act[2]:
                facility = cur_act[2].split('-')[1]
                if cur_act[0] == 'Inform':
                    cur_act[2] = cur_act[2].split('-')[0] + '+' + cur_act[3]
                elif cur_act[0] == 'Request':
                    cur_act[2] = cur_act[2].split('-')[0]
            if cur_act[0] == 'Select':
                cur_act[2] = '源领域+' + cur_act[3]
            intent = '+'.join(cur_act[:-1])
            if '+'.join(cur_act) == 'Inform+景点+门票+免费' or cur_act[-1] == '无':
                intent = '+'.join(cur_act)
            intent_list.append(intent)

            # content replacement
            if (act[0] in ['Inform', 'Recommend'] or '酒店设施' in intent) and not intent.endswith('无'):
                if act[3] in content or (facility and facility in content):
                    intent_frequency[intent] += 1

                    # value to be replaced
                    if '酒店设施' in intent:
                        value = facility
                    else:
                        value = act[3]

                    # placeholder
                    placeholder = '[' + intent + ']'
                    placeholder_one = '[' + intent + '1]'
                    placeholder_with_number = '[' + intent + str(intent_frequency[intent]) + ']'

                    if intent_frequency[intent] > 1:
                        content = content.replace(placeholder, placeholder_one)
                        content = content.replace(value, placeholder_with_number)
                    else:
                        content = content.replace(value, placeholder)
                else:
                    usable = False

        # multi-intent name
        try:
            intent_list = sorted(intent_list, key=functools.cmp_to_key(cmp_intent))
        except:
            print(round['content'])
        multi_intent = '*'.join(intent_list)
        if usable:
            if round['role'] == 'usr':
                user_multi_intent_dict[multi_intent].append(content)
            else:
                sys_multi_intent_dict[multi_intent].append(content)

print('Length of user_multi_intent_dict: %d' % len(user_multi_intent_dict))
print('Length of sys_multi_intent_dict: %d' % len(sys_multi_intent_dict))

with open('auto_user_template_nlg.json', 'w', encoding='utf-8') as f:
    json.dump(user_multi_intent_dict, f, indent=4, sort_keys=True, ensure_ascii=False)

with open('auto_system_template_nlg.json', 'w', encoding='utf-8') as f:
    json.dump(sys_multi_intent_dict, f, indent=4, sort_keys=True, ensure_ascii=False)


