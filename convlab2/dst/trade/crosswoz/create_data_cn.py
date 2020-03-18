import json
from os.path import join, abspath, dirname
import jieba
import sys
import copy

root_path = dirname(abspath(__file__))
data_path = join(root_path, 'data')
crosswoz_path = join(data_path, 'crosswoz')
userdict_path = join(crosswoz_path, 'user.dict')

jieba.load_userdict(userdict_path)

# build_ontology
def build_ontology():
    splits = ['train', 'val', 'test']
    ontology = {}

    def invalid_value(v):
        if v is None or v == '':
            return True
        return False

    def push(dic, key, val):
        if key not in dic:
            dic[key] = [val]
        else:
            lst = dic[key]
            if val not in lst:
                dic[key].append(val)

    def add_val(domain, slot, value):
        if slot == 'selectedResults': return
        if type(value) is str:
            if not invalid_value(value): push(ontology, domain+'-'+slot, sentseg(value))
        elif type(value) is list:
            for v in value:
                if not invalid_value(v): push(ontology, domain+'-'+slot, sentseg(v))
        else:
            raise Exception()

    for item in splits:
        print(f'split: {item}')
        filename = join(crosswoz_path, f'{item}.json')
        data = json.load(open(filename, encoding='utf-8'))
        sze = len(data)
        for idx, id in enumerate(data.keys()):
            if idx % 100 == 0:
                print(f'\t{idx}/{sze}')
            session = data[id]
            # from goal
            goal = session['goal']
            for _, domain, slot, value, _ in goal:
                add_val(domain, slot, value)
            for turn in session['messages']:
                acts = turn['dialog_act']
                # from dialogue act
                for _, domain, slot, value in acts:
                    add_val(domain, slot, value)
                # from dialogue state
                if 'sys_state_init' in turn:
                    sys_state_init = turn['sys_state_init']
                    for domain in sys_state_init:
                        for slot in sys_state_init[domain]:
                            add_val(domain, slot, sys_state_init[domain][slot])
    json.dump(ontology, open(join(crosswoz_path, 'ontology.json'), 'w+', encoding='utf-8'), indent=2, ensure_ascii=False)

# word segment
def segdata():
    splits = ['test', 'train', 'val']
    for item in splits:
        print(f'seg: {item}')
        name = join(crosswoz_path, f'{item}.json')
        data = json.load(open(name))
        for idx, sess_id in enumerate(data.keys()):
            if idx % 100 == 0:
                print(f'\t{idx}/{len(data)}')
            sess = data[sess_id]
            prev_state, prev_init_state, user_sent = None, None, None
            for message in sess['messages']:
                if message['role'] == 'usr':
                    seg_user_sent = sentseg(message['content'])
                    message['content'] = seg_user_sent
                    user_sent = message['content']
                elif message['role'] == 'sys':
                    message['content'] = sentseg(message['content'])
                    seg_state(message['sys_state'], prev_state, user_sent, message['content'])
                    seg_state(message['sys_state_init'], prev_init_state, user_sent, message['content'])
                    prev_init_state = copy.deepcopy(message['sys_state_init'])
                    prev_state = copy.deepcopy(message['sys_state'])
        json.dump(data, open(join(crosswoz_path, f'{item}_seg.json'), 'w+'), indent=2, ensure_ascii=False)


def seg_state(state, prev_state, user_sent, sys_sent):
    def equal_value(a, b):
        if a == b or a == ''.join(b.split()):
            return True
        return False
    def in_sent(a, b):
        if a in b or a in ''.join(b.split()):
            return True
        return False
    def in_uttr(a, b, c):
        if in_sent(a, b) or in_sent(a, c):
            return True
        return False
    def get_seg_val(a, b, c):
        if in_sent(a, b):
            return find_seg_from_sent(a, b)
        elif in_sent(a, c):
            return find_seg_from_sent(a, c)
        else:
            raise Exception(a, b, c)
    def find_seg_from_sent(a, b):
        if a in b:
            return a
        else:
            b_ = ''.join(b.split())
            fake_idx = b_.find(a)
            assert fake_idx > -1, b_ + '___' + a
            new_a = ''
            tag = False
            count = 0
            for idx in range(len(b)):
                if b[idx:idx+1] != ' ':
                    count += 1
                if fake_idx + len(a) >= count > fake_idx:
                    tag = True
                if count > fake_idx + len(a):
                    tag = False
                if tag:
                    new_a += b[idx:idx+1]
            new_a = new_a.strip()
            return new_a

    for domain in state.keys():
        domain_state = state[domain]
        for slot in domain_state.keys():
            if slot == 'selectedResults':
                pass
            else:
                assert type(domain_state[slot] is str), domain_state
                val = state[domain][slot]
                if prev_state is None or not equal_value(val, prev_state[domain][slot]):
                    if in_uttr(val, user_sent, sys_sent):
                        seg_val = get_seg_val(val, user_sent, sys_sent)
                        domain_state[slot] = seg_val
                    else:
                        seg_val = sentseg(val)
                        domain_state[slot] = seg_val
                elif prev_state is not None and equal_value(val, prev_state[domain][slot]):
                        seg_val = prev_state[domain][slot]
                        domain_state[slot] = seg_val
                else:
                    raise Exception()



init_turn = {
    'system_transcript': '',
    'turn_idx': 0,
    'system_acts': []
}
def get_init_turn():
    return copy.deepcopy(init_turn)

def get_user_domains(user_act, tag):
    count = {}
    for _, domain, _, _ in user_act:
        if tag:
            if domain in ['thank', 'greet', 'welcome']: continue
        if domain not in count:
            count[domain] = 0
        count[domain] += 1
    max_domain, max_count = None, -1
    for d, c in count.items():
        if c > max_count:
            max_count = c
            max_domain = d
    return max_domain

def get_belief_state(state):
    new_state = []
    for domain in state:
        for slot in state[domain]:
            if slot == 'selectedResults': continue
            val = state[domain][slot]
            if val is not None and val != '':
                new_state.append({'slots': [[f'{domain}-{slot}', val]]})
    return new_state

# convert data
def convert_all(mode='init'):
    splits = ['train', 'val', 'test']
    for item in splits:
        name = join(crosswoz_path, f'{item}_seg.json')
        convert(name, item, mode)

def convert(name, item, mode):
    all_domains = set()
    dat = json.load(open(name))
    data = []
    sze = len(dat)
    print(f'split: {item}')
    for idx, sess_id in enumerate(dat.keys()):
        if idx % 100 == 0:
            print(f'\t{idx}/{sze}')
        turns = []
        turn_info = get_init_turn()
        domains = set()
        for turn_idx, msg in enumerate(dat[sess_id]['messages']):
            if msg['role'] == 'usr':
                turn_info['turn_idx'] = turn_idx // 2
                turn_info['transcript'] = msg['content']
                if turn_idx + 1 < len(dat[sess_id]['messages']):
                    if mode == 'init':
                        turn_info['belief_state'] = get_belief_state(dat[sess_id]['messages'][turn_idx+1]['sys_state_init'])
                    else:
                        turn_info['belief_state'] = get_belief_state(dat[sess_id]['messages'][turn_idx + 1]['sys_state'])
                    d = get_user_domains(msg['dialog_act'], True)
                    if d is None:
                        if len(turns) > 0:
                            d = turns[-1]['domain']
                        else:
                            d = get_user_domains(msg['dialog_act'], False)
                    if d is None:
                        d = 'greet'
                    assert d is not None
                    all_domains.add(d)
                    turn_info['domain'] = d
                    domains.add(turn_info['domain'])
                    turns.append(copy.deepcopy(turn_info))
            else:
                turn_info = {}
                turn_info['system_transcript'] = msg['content']
        sess = {
            'dialogue_idx': sess_id,
            'domains': list(domains),
            'dialogue': turns
        }
        data.append(sess)
    # model
    print(f'all_domains: {list(all_domains)}')
    if mode == 'init':
        json.dump(data, open(join(crosswoz_path, f'{item}_dials.json'), 'w+'), ensure_ascii=False, indent=2)
    else:
        json.dump(data, open(join(crosswoz_path, f'{item}_dials_processed.json'), 'w+'), ensure_ascii=False, indent=2)


def sentseg(sent):
    sent = sent.replace('\t', ' ')
    sent = ' '.join(sent.split())
    tmp = " ".join(jieba.cut(sent))
    return ' '.join(tmp.split())

def convert_prediction(mode):
    path = 'data/crosswoz/all_prediction_TRADE_init.json'
    if mode == 'processed':
        path = 'data/crosswoz/all_prediction_TRADE.json'
    data = json.load(open(path))
    target_path = 'prediction.json'
    if mode == 'processed':
        target_path = 'prediction_processed.json'
    json.dump(data, open(join(crosswoz_path, target_path), 'w+'), ensure_ascii=False, indent=2)

init_belief_state = {
        "景点": {
            "名称": "",
            "门票": "",
            "游玩时间": "",
            "评分": "",
            "周边景点": "",
            "周边餐馆": "",
            "周边酒店": "",
            "selectedResults": []
        },
        "餐馆": {
            "名称": "",
            "推荐菜": "",
            "人均消费": "",
            "评分": "",
            "周边景点": "",
            "周边餐馆": "",
            "周边酒店": "",
            "selectedResults": [
                "鲜鱼口老字号美食街"
            ]
        },
        "酒店": {
            "名称": "",
            "酒店类型": "",
            "酒店设施": "",
            "价格": "",
            "评分": "",
            "周边景点": "",
            "周边餐馆": "",
            "周边酒店": "",
            "selectedResults": []
        },
        "地铁": {
            "出发地": "",
            "目的地": "",
            "selectedResults": []
        },
        "出租": {
            "出发地": "",
            "目的地": "",
            "selectedResults": []
        }
    }

def prediction_to_original(patha='data/crosswoz/prediction_fine.json', pathaa='data/crosswoz/prediction_processed_fine.json', pathb='data/crosswoz/test.json'):
    pred = json.load(open(patha))
    pred_p = json.load(open(pathaa))
    data = json.load(open(pathb))
    for idx, sess_id in enumerate(pred.keys()):
        if idx % 50 == 0:
            print(f'\t{idx}/{len(pred)}')
        sess = pred[sess_id]
        for turn_id, item in sess.items():
            turn_id = int(turn_id)
            pred_bs_ptr = item['pred_bs_ptr']
            real_turn_id = 2*turn_id+1
            target_turn = data[sess_id]['messages'][real_turn_id]
            assert target_turn['role'] == 'sys'
            new_state = copy.deepcopy(init_belief_state)
            for dsv in pred_bs_ptr:
                domain, slot, val = dsv.split('-', 2)
                val = ''.join(val.split())
                new_state[domain][slot] = val
            key_name = 'sys_state_init_pred'
            target_turn[key_name] = new_state
    for idx, sess_id in enumerate(pred_p.keys()):
        if idx % 50 == 0:
            print(f'\t{idx}/{len(pred)}')
        sess = pred_p[sess_id]
        for turn_id, item in sess.items():
            turn_id = int(turn_id)
            pred_bs_ptr = item['pred_bs_ptr']
            real_turn_id = 2*turn_id+1
            target_turn = data[sess_id]['messages'][real_turn_id]
            assert target_turn['role'] == 'sys'
            new_state = copy.deepcopy(init_belief_state)
            for dsv in pred_bs_ptr:
                domain, slot, val = dsv.split('-', 2)
                val = ''.join(val.split())
                new_state[domain][slot] = val
            key_name = 'sys_state_pred'
            target_turn[key_name] = new_state
    filename = pathb[:-5]+'_with_pred.json'
    json.dump(data, open(filename, 'w+'), ensure_ascii=False, indent=4)

slot_temp = ['景点-门票', '景点-评分', '餐馆-名称', '酒店-价格', '酒店-评分', '景点-名称', '景点-地址', '景点-游玩时间',
             ' 餐馆-营业时间', '餐馆-评分', '酒店-名称', '酒店-周边景点', '酒店-酒店设施-叫醒服务', '酒店-酒店类型',
             '餐馆-人 均消费', '餐馆-推荐菜', '酒店-酒店设施', '酒店-电话', '景点-电话', '餐馆-周边餐馆', '餐馆-电话',
             '餐馆-none', '餐馆-地址', '酒店-酒店设施-无烟房', '酒店-地址', '景点-周边景点', '景点-周边酒店', '出租-出发地',
             '出租-目的地', '地铁-出发地', '地铁-目的地', '景点-周边餐馆', '酒店-周边餐馆', '出租-车型', '餐馆-周边景点',
             '餐馆-周边酒店', '地铁-出发地附近地铁站', '地铁-目的地附近地铁站', '景点-none', '酒店-酒店设施-商务中心',
             '餐馆-源领域', '酒店-酒店设施-中式餐厅', '酒店-酒店设施-接站服务', '酒店-酒店设施-国际长途电话', '酒店-酒店设施-吹风机',
             '酒店-酒店设施-会议室', '酒店-源领域', '酒店-none', '酒店-酒店设施-宽带上网', '酒店-酒店设施-看护小孩服务',
             '酒店-酒店设 施-酒店各处提供wifi', '酒店-酒店设施-暖气', '酒店-酒店设施-spa', '出租-车牌', '景点-源领域',
             '酒店-酒店设施-行 李寄存', '酒店-酒店设施-西式餐厅', '酒店-酒店设施-酒吧', '酒店-酒店设施-早餐服务',
             '酒店-酒店设施-健身房', '酒 店-酒店设施-残疾人设施', '酒店-酒店设施-免费市内电话', '酒店-酒店设施-接待外宾',
             '酒店-酒店设施-部分房间提供wifi', '酒店-酒店设施-洗衣服务', '酒店-酒店设施-租车',
             '酒店-酒店设施-公共区域和部分房间提供wifi', '酒店-酒店设施-24小时热水', '酒店-酒店设施-温泉', '酒店-酒店设施-桑拿',
             '酒店-酒店设施-收费停车位', '酒店-周边酒店', '酒店-酒 店设施-接机服务', '酒店-酒店设施-所有房间提供wifi',
             '酒店-酒店设施-棋牌室', '酒店-酒店设施-免费国内长途电话', '酒店-酒店设施-室内游泳池', '酒店-酒店设施-早餐服务免费',
             '酒店-酒店设施-公共区域提供wifi', '酒店-酒店设施-室外 游泳池']


def compute_acc(gold, pred, slot_temp):
    miss_gold = 0
    miss_slot = []
    for g in gold:
        if g not in pred:
            miss_gold += 1
            miss_slot.append(g.rsplit("-", 1)[0])
    wrong_pred = 0
    for p in pred:
        if p not in gold and p.rsplit("-", 1)[0] not in miss_slot:
            wrong_pred += 1
    ACC_TOTAL = len(slot_temp)
    ACC = len(slot_temp) - miss_gold - wrong_pred
    ACC = ACC / float(ACC_TOTAL)
    return ACC

def compute_prf(gold, pred):
    TP, FP, FN = 0, 0, 0
    if len(gold)!= 0:
        count = 1
        for g in gold:
            if g in pred:
                TP += 1
            else:
                FN += 1
        for p in pred:
            if p not in gold:
                FP += 1
        precision = TP / float(TP+FP) if (TP+FP)!=0 else 0
        recall = TP / float(TP+FN) if (TP+FN)!=0 else 0
        F1 = 2 * precision * recall / float(precision + recall) if (precision+recall)!=0 else 0
    else:
        if len(pred)==0:
            precision, recall, F1, count = 1, 1, 1, 1
        else:
            precision, recall, F1, count = 0, 0, 0, 1
    return F1, recall, precision, count

def evaluate_metrics(all_prediction, from_which, slot_temp):
    total, turn_acc, joint_acc, F1_pred, F1_count = 0, 0, 0, 0, 0
    for d, v in all_prediction.items():
        for t in range(len(v)):
            cv = v[str(t)]
            if set(cv["turn_belief"]) == set(cv[from_which]):
                joint_acc += 1
            total += 1

            # Compute prediction slot accuracy
            temp_acc = compute_acc(set(cv["turn_belief"]), set(cv[from_which]), slot_temp)
            turn_acc += temp_acc

            # Compute prediction joint F1 score
            temp_f1, temp_r, temp_p, count = compute_prf(set(cv["turn_belief"]), set(cv[from_which]))
            F1_pred += temp_f1
            F1_count += count

    joint_acc_score = joint_acc / float(total) if total != 0 else 0
    turn_acc_score = turn_acc / float(total) if total != 0 else 0
    F1_score = F1_pred / float(F1_count) if F1_count != 0 else 0
    return joint_acc_score, F1_score, turn_acc_score

def evaluate_after_select_metrics(all_prediction, data, from_which, slot_temp):
    total, turn_acc, joint_acc, F1_pred, F1_count = 0, 0, 0, 0, 0
    for d, v in all_prediction.items():
        for t in range(len(v)):
            real_turn_id = 2 * t + 1
            cv = v[str(t)]
            # check if after select
            if real_turn_id <= 0:
                continue
            target_turn = data[d]['messages'][real_turn_id-1]
            assert target_turn['role'] == 'usr'
            has_select = False
            for intent, _, _, _ in target_turn['dialog_act']:
                if intent == 'Select':
                    has_select = True
                    break
            if not has_select:
                continue

            if set(cv["turn_belief"]) == set(cv[from_which]):
                joint_acc += 1
            total += 1

    joint_acc_score = joint_acc / float(total) if total != 0 else 0
    return joint_acc_score, joint_acc, total

def evaluate(mode):
    source_path = 'data/crosswoz/prediction_fine.json'
    # source_path = 'data/crosswoz/prediction.json'
    if mode == 'processed':
        source_path = 'data/crosswoz/prediction_processed_fine.json'
    dat = json.load(open(source_path))

    original_path = 'data/crosswoz/test.json'
    original_data = json.load(open(original_path))

    for tpe in ['S', 'M', 'M+T', 'CM', 'CM+T', 'ALL']:
        type_data = type_filter(dat, tpe, original_data)
        joint, f1, turn = evaluate_metrics(type_data, 'pred_bs_ptr', slot_temp)
        print(f'mode: {mode}, type: {tpe}, size: {len(type_data)} joint acc: {joint}, F1: {f1}, Turn: {turn}')

def evaluate_select(mode):
    source_path = 'data/crosswoz/prediction_fine.json'
    # source_path = 'data/crosswoz/prediction.json'
    if mode == 'processed':
        source_path = 'data/crosswoz/prediction_processed_fine.json'
    dat = json.load(open(source_path))

    original_path = 'data/crosswoz/test.json'
    original_data = json.load(open(original_path))

    for tpe in ['S', 'M', 'M+T', 'CM', 'CM+T', 'ALL']:
        type_data = type_filter(dat, tpe, original_data)
        joint, fenzi, fenmu = evaluate_after_select_metrics(type_data, original_data, 'pred_bs_ptr', slot_temp)
        print(f'mode: {mode}, type: {tpe}, size: {len(type_data)} joint acc: {joint}, Corrent Turn #: {fenzi}, Total Turn #: {fenmu}')

def type_filter(data, tpe, original_data):
    if tpe == 'ALL': return data
    tpe_dict = {
        'S': "单领域",
        'M': "独立多领域",
        'M+T': "独立多领域+交通",
        'CM': "不独立多领域",
        'CM+T': "不独立多领域+交通",
        'ALL': "全部"
    }
    real_type = tpe_dict[tpe]
    filtered_data = {}
    for sess_id in data.keys():
        sess_type = original_data[sess_id]['type']
        if sess_type == real_type:
            filtered_data[sess_id] = copy.deepcopy(data[sess_id])
    return filtered_data




if __name__ == '__main__':
    # print(sentseg("江州市长江大桥参加了长江大桥的通车仪式"))
    segdata()
    build_ontology()
    convert_all(mode='init')
    convert_prediction('processed')
    evaluate('processed')
    evaluate_select('init')
    prediction_to_original()