import json
import random
import os
from pprint import pprint
import functools
import copy
from collections import defaultdict
import zipfile
from convlab2.nlg import NLG


def read_json(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)


def read_zipped_json(filepath, filename):
    archive = zipfile.ZipFile(filepath, 'r')
    return json.load(archive.open(filename))


class TemplateNLG(NLG):
    def __init__(self, is_user, mode="auto_manual"):
        # super().__init__()
        self.is_user = is_user
        self.mode = mode
        if is_user:
            self.role = 'usr'
        else:
            self.role = 'sys'
        template_dir = os.path.dirname(os.path.abspath(__file__))
        # multi-intent
        self.auto_user_template = read_json(os.path.join(template_dir, 'auto_user_template_nlg.json'))
        self.auto_system_template = read_json(os.path.join(template_dir, 'auto_system_template_nlg.json'))
        # single-intent
        self.manual_user_template = read_json(os.path.join(template_dir, 'manual_user_template_nlg.json'))
        self.manual_system_template = read_json(os.path.join(template_dir, 'manual_system_template_nlg.json'))

    def generate(self, dialog_act):
        """

        :param dialog_act: [["Request", "景点", "名称", ""], ["Inform", "景点", "门票", "免费"], ...]
        :return: a sentence
        """
        dialog_act = [[str(x[0]), str(x[1]), str(x[2]), str(x[3]).lower()] for x in dialog_act]
        # print(dialog_act)
        dialog_act = copy.deepcopy(dialog_act)
        mode = self.mode
        try:
            is_user = self.is_user
            if mode == 'manual':
                if is_user:
                    template = self.manual_user_template
                else:
                    template = self.manual_system_template

                return self._manual_generate(copy.deepcopy(dialog_act), template)
            elif mode == 'auto':
                if is_user:
                    template = self.auto_user_template
                else:
                    template = self.auto_system_template

                return self._auto_generate(copy.deepcopy(dialog_act), template)
            elif mode == 'auto_manual':
                if is_user:
                    template1 = self.auto_user_template
                    template2 = self.manual_user_template
                else:
                    template1 = self.auto_system_template
                    template2 = self.manual_system_template
                try:
                    res = self._auto_generate(copy.deepcopy(dialog_act), template1)
                except:
                    res = self._manual_generate(copy.deepcopy(dialog_act), template2)
                return res

            else:
                raise Exception("\n\nInvalid mode! available mode: auto, manual, auto_manual")

        except Exception as e:
            print('\n\nError in processing:')
            pprint(copy.deepcopy(dialog_act))
            return ''
            # raise e

    def _postprocess(self, sen, last_sen=False):
        sen = sen.strip('。.，, ')
        if sen[-1] not in ['!', '?', '！', '？']:
            if last_sen:
                sen += '。'
            else:
                sen += '，'
        return sen

    def _value_replace(self, sentences, dialog_act):
        dialog_act = copy.deepcopy(dialog_act)
        intent_frequency = defaultdict(int)
        for act in dialog_act:
            intent = self._prepare_intent_string(copy.copy(act))
            intent_frequency[intent] += 1
            if intent_frequency[intent] > 1:  # if multiple same intents...
                intent += str(intent_frequency[intent])

            if '酒店设施' in intent:
                try:
                    sentences = sentences.replace('[' + intent + ']', act[2].split('-')[1])
                    sentences = sentences.replace('[' + intent + '1]', act[2].split('-')[1])
                except Exception as e:
                    print('Act causing problem in replacement:')
                    pprint(act)
                    raise e
            if act[0] == 'Inform' and act[3] == "无":
                sentences = sentences.replace('[主体]', act[1])
                sentences = sentences.replace('[属性]', act[2])
            sentences = sentences.replace('[' + intent + ']', act[3])
            sentences = sentences.replace('[' + intent + '1]', act[3])  # if multiple same intents and this is 1st

        if '[' in sentences and ']' in sentences:
            print('\n\nValue replacement not completed!!! Current sentence: %s' % sentences)
            print('Current DA:')
            pprint(dialog_act)
            # raise Exception
        return sentences

    def _multi_same_intent_process(self, base_intent: str, repetition: int):
        """

        :param base_intent: e.g. "Inform+餐馆+推荐菜"
        :param frequency: e.g. 2
        :return:e.g. "Inform+餐馆+推荐菜1+推荐菜2"
        """
        if repetition == 1:
            return base_intent
        elif repetition > 1:
            try:
                return base_intent + '1+' + '+'.join([base_intent.split('+')[-1] + str(i) for i in range(2, repetition + 1)])
            except:
                print(base_intent, repetition)
        else:
            raise Exception("Repetition should take value in {1, 2, ...}")

    def _manual_generate(self, dialog_act, template):
        dialog_act = copy.deepcopy(dialog_act)
        intent_list = self._prepare_intent_string_list(copy.deepcopy(dialog_act))
        sentences = ''
        while intent_list:
            intent = intent_list.pop(0)

            # "Recommend+酒店+名称1+名称2+名称3+名称4"等：
            if intent not in template.keys() and '1' in intent:
                base_intent = '+'.join(intent.split('+')[:3]).strip('1')
                repetition = len(intent.split('+')) - 2 - 1  # times of repetition - 1
                while self._multi_same_intent_process(base_intent, repetition) not in template.keys() and repetition >= 1:
                    repetition -= 1
                if len(intent.split('+')) - 2 - repetition >= 1:
                    intent_list = [self._multi_same_intent_process(base_intent,
                                                                   len(intent.split('+')) - 2 - repetition)] + intent_list
                intent = self._multi_same_intent_process(base_intent, repetition)
            elif 'Inform' in intent and '无' in intent:
                intent = 'Inform+主体+属性+无'

            sentence = random.choice(template[intent])
            sentence = self._postprocess(sentence, intent_list == [])
            sentences += sentence
            # slot replacement:
            sentences = self._value_replace(sentences, copy.deepcopy(dialog_act))
        return sentences

    def _auto_generate(self, dialog_act, template):
        dialog_act = copy.deepcopy(dialog_act)
        intent_list = self._prepare_intent_string_list(copy.deepcopy(dialog_act))
        multi_intent = '*'.join(intent_list)
        try:
            sentences = random.choice(template[multi_intent])
            # slot replacement:
            sentences = self._value_replace(sentences, copy.deepcopy(dialog_act))

        except Exception as e:  # todo address the error
            # if multi_intent not in template.keys():
            #     print('\n\nIntent combination not found in auto-generation templates: \n\t%s. \nTurned into manual mode.' % multi_intent)
            # print(repr(e))
            raise e
        return sentences

    def _cmp_intent(self, intent1: str, intent2: str):
        role = self.role
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
            raise AssertionError
        return intent_order[role].index(intent1) - intent_order[role].index(intent2)

    def _prepare_intent_string_list(self, dialog_act):
        """

        :param dialog_act: [["Request", "景点", "名称", ""], ["Inform", "景点", "门票", "免费"], ...]
        :return: a sorted list of intent strings: ["Inform+景点+门票+免费", "Request+景点+名称", ...]
        """
        dialog_act = copy.deepcopy(dialog_act)
        intent_frequency = defaultdict(int)
        intent_list = []

        for act in copy.deepcopy(dialog_act):
            cur_act = copy.copy(act)
            intent = self._prepare_intent_string(cur_act)
            intent_list.append(intent)
            intent_frequency[intent] += 1

        if self.mode == 'manual':
            # for intents like "Inform+景点+周边酒店1+周边酒店2+周边酒店3+周边酒店4":
            for intent in intent_frequency.keys():
                if intent_frequency[intent] > 1:
                    if 'Recommend' in intent or '名称' in intent or '推荐菜' in intent or '周边' in intent:
                        new_intent = intent + '1+' + '+'.join([
                            intent.split('+')[-1] + str(k) for k in range(2, intent_frequency[intent] + 1)])
                        intent_frequency[new_intent] = 1
                        del intent_frequency[intent]
            intent_list = sorted(intent_frequency.keys(), key=functools.cmp_to_key(self._cmp_intent))
            return copy.copy(intent_list)
        else:
            intent_list = sorted(intent_list, key=functools.cmp_to_key(self._cmp_intent))
            return copy.copy(intent_list)

    def _prepare_intent_string(self, cur_act):
        """
        Generate the intent form **to be used in selecting templates** (rather than value replacement)
        :param cur_act: one act list
        :return: one intent string
        """
        cur_act = copy.deepcopy(cur_act)
        if cur_act[0] == 'Inform' and '酒店设施' in cur_act[2]:
            cur_act[2] = cur_act[2].split('-')[0] + '+' + cur_act[3]
        elif cur_act[0] == 'Request' and '酒店设施' in cur_act[2]:
            cur_act[2] = cur_act[2].split('-')[0]
        if cur_act[0] == 'Select':
            cur_act[2] = '源领域+' + cur_act[3]
        try:
            if '+'.join(cur_act) == 'Inform+景点+门票+免费':
                intent = '+'.join(cur_act)
            # "Inform+景点+周边酒店+无"
            elif cur_act[3] == '无':
                intent = '+'.join(cur_act)
            else:
                intent = '+'.join(cur_act[:-1])
        except Exception as e:
            print('Act causing error:')
            pprint(cur_act)
            raise e
        return intent


def example():
    data_dir = '../../../../data/crosswoz/'
    train_data = read_zipped_json(os.path.join(data_dir, 'test.json.zip'), 'test.json')
    messages = [d["messages"] for d in train_data.values()]
    for i in range(100):
        for message in random.choices(messages):
            for r in message:
                dialog_act = r['dialog_act']

                if not dialog_act:
                    continue  # DA = []

                print('\n\nCurrent role:', r['role'])
                print('Current DA:')
                pprint(dialog_act)

                # system model for manual, auto, auto_manual
                nlg_sys_manual = TemplateNLG(is_user=r['role'] == 'usr', mode='manual')
                nlg_sys_auto = TemplateNLG(is_user=r['role'] == 'usr', mode='auto')
                nlg_sys_auto_manual = TemplateNLG(is_user=r['role'] == 'usr', mode='auto_manual')

                # generate
                try:
                    print('manual      : ', nlg_sys_manual.generate(dialog_act))
                    print('auto        : ', nlg_sys_auto.generate(dialog_act))
                    print('auto_manual : ', nlg_sys_auto_manual.generate(dialog_act))
                except Exception as e:
                    print("Generation failure.")
                    print(repr(e))


if __name__ == '__main__':
    nlg = TemplateNLG(is_user=False)
    print(nlg.generate([['Inform', '地铁', '目的地', '云峰山'], ['Request', '地铁', '出发地', '']]))
