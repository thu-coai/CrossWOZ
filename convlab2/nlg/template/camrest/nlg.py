import json
import random
import os
from pprint import pprint
from convlab2.nlg import NLG


class TemplateNLG(NLG):
    def __init__(self, is_user, mode="manual"):
        """
        Args:
            is_user:
                if dialog_act from user or system
            mode:
                - `auto`: templates extracted from data without manual modification, may have no match;

                - `manual`: templates with manual modification, sometimes verbose;

                - `auto_manual`: use auto templates first. When fails, use manual templates.

                both template are dict, *_template[dialog_act][slot] is a list of templates.
        """
        super().__init__()
        self.is_user = is_user
        self.mode = mode
        template_dir = os.path.dirname(os.path.abspath(__file__))
        self.auto_user_template = self.read_json(os.path.join(template_dir, 'auto_user_template_nlg.json'))
        self.auto_system_template = self.read_json(os.path.join(template_dir, 'auto_system_template_nlg.json'))
        self.manual_user_template = self.read_json(os.path.join(template_dir, 'manual_user_template_nlg.json'))
        self.manual_system_template = self.read_json(os.path.join(template_dir, 'manual_system_template_nlg.json'))

    def generate(self, dialog_acts):
        """NLG for Multiwoz dataset

        Args:
            dialog_acts:
                {da1:[[slot1,value1],...], da2:...}
        Returns:
            generated sentence
        """
        action = {}
        for intent, slot, value in dialog_acts:
            k = intent
            action.setdefault(k, [])
            action[k].append([slot, value])
        dialog_acts = action
        mode = self.mode
        try:
            is_user = self.is_user
            if mode == 'manual':
                if is_user:
                    template = self.manual_user_template
                else:
                    template = self.manual_system_template

                return self._manual_generate(dialog_acts, template)

            elif mode == 'auto':
                if is_user:
                    template = self.auto_user_template
                else:
                    template = self.auto_system_template

                return self._auto_generate(dialog_acts, template)

            elif mode == 'auto_manual':
                if is_user:
                    template1 = self.auto_user_template
                    template2 = self.manual_user_template
                else:
                    template1 = self.auto_system_template
                    template2 = self.manual_system_template

                return self._auto_manual_generate(dialog_acts, template1, template2)

            else:
                raise Exception("Invalid mode! available mode: auto, manual, auto_manual")
        except Exception as e:
            print('Error in processing:')
            pprint(dialog_acts)
            raise e

    def _auto_manual_generate(self, dialog_acts, temp_auto, temp_manual):
        sentences = ''
        for intent in ['nooffer', 'inform', 'request']:
            if intent not in dialog_acts.keys():
                continue

            sentence = self._generate_single_intent_auto(intent, dialog_acts[intent], temp_auto)
            if sentence == 'None':
                sentence = self._generate_single_intent_manual(intent, dialog_acts[intent], temp_manual)

            sentences += ' ' + sentence

        return sentences.strip()

    def _manual_generate(self, dialog_acts, template):
        sentences = ''
        for intent in ['nooffer', 'inform', 'request']:
            if intent not in dialog_acts.keys():
                continue

            sentences += ' ' + self._generate_single_intent_manual(intent, dialog_acts[intent], template)

        return sentences.strip()

    def _auto_generate(self, dialog_acts, template):
        sentences = ''
        for intent in ['nooffer', 'inform', 'request']:
            if intent not in dialog_acts.keys():
                continue

            sentence = self._generate_single_intent_auto(intent, dialog_acts[intent], template)
            if sentence == 'None':
                return 'None'

            sentences += ' ' + sentence

        return sentences.strip()

    def _generate_single_intent_manual(self, intent, slot_value_pairs, template):
        sentences = ''
        if 'request' == intent:
            for slot, value in slot_value_pairs:
                if intent not in template or slot not in template[intent]:
                    sentence = ' What is the %s ? ' % slot
                else:
                    sentence = random.choice(template[intent][slot])
                    sentence = self._postprocess(sentence)
                sentences += sentence
        elif intent in ['nooffer', 'inform']:
            for slot, value in slot_value_pairs:
                if intent in template and slot in template[intent] and value != 'dontcare':
                    sentence = random.choice(template[intent][slot])
                    sentence = sentence.replace('#%s-%s#' % (intent.upper(), slot.upper()), str(value), 1)
                else:
                    if intent == 'inform':
                        if value != 'dontcare':
                            sentence = ' The %s is %s . ' % (slot, str(value))
                        else:
                            sentence = self._generate_notcare_text(slot)
                    else:
                        sentence = ' We do\'nt have a place that matches those qualities. Can you try something else? '
                        sentences = sentence
                        break
                sentence = self._postprocess(sentence)
                sentences += sentence

        return sentences.strip()

    def _generate_single_intent_auto(self, intent, slot_value_pairs, template):
        key = ''
        for s, v in sorted(slot_value_pairs, key=lambda x: x[0]):
            key += s + ';'
        if intent in template and key in template[intent]:
            sentence = random.choice(template[intent][key])
            if intent != 'request':
                for s, v in sorted(slot_value_pairs, key=lambda x: x[0]):
                    if v == 'dontcare':
                        return 'None'
                    if v != 'none':
                        sentence = sentence.replace('#%s-%s#' % (intent.upper(), s.upper()), v, 1)
            return self._postprocess(sentence)
        else:
            return 'None'

    @staticmethod
    def _postprocess(sen):
        sen_strip = sen.strip()
        sen = ''.join([val.capitalize() if i == 0 else val for i, val in enumerate(sen_strip)])
        if sen and sen[-1] != '?' and sen[-1] != '.':
            sen += '.'
        sen += ' '
        return sen

    @staticmethod
    def _generate_notcare_text(slot):
        text = 'Don\'t care about %s . ' % slot
        if slot == 'pricerange':
            text = 'Any price range is fine.'
        elif slot == 'area':
            text = 'It doesn\'t matter about area.'
        elif slot == 'food':
            text = 'I don\'t care about food kind.'
        return ' ' + text

    @staticmethod
    def read_json(filename):
        with open(filename, 'r') as f:
            return json.load(f)


def test():
    data_file = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, os.pardir, 'data/camrest/train.json')
    data = TemplateNLG.read_json(data_file)

    def test_role(role):
        model_a = TemplateNLG(role == 'user', 'auto')
        model_m = TemplateNLG(role == 'user', 'manual')
        model_am = TemplateNLG(role == 'user', 'auto_manual')

        num_of_all = 0
        num_of_non = 0
        for session in data:
            qas = session['dial']
            for qa in qas:
                if role == 'system':
                    text = qa['sys']['sent']
                    da = qa['sys']['dialog_act']
                else:
                    text = qa['usr']['transcript']
                    da = qa['usr']['dialog_act']

                gen_utt_a = model_a.generate(da)
                gen_utt_m = model_m.generate(da)
                gen_utt_am = model_am.generate(da)
                num_of_all += 1
                if gen_utt_a == 'None':
                    num_of_non += 1

                print('DA  : ' + str(da))
                print('Real: ' + text)
                print('A   : ' + gen_utt_a)
                print('M   : ' + gen_utt_m)
                print('AM  : ' + gen_utt_am)
                print('--------------------------------')

        print('%d / %d = %.4f' % (num_of_non, num_of_all, num_of_non * 1.0 / num_of_all))

    test_role('system')
    test_role('user')

def example():
    # dialog act
    dialog_acts = [['inform', 'pricerange', 'cheap'], ['inform', 'area', 'west']]
    print(dialog_acts)

    # system model for manual, auto, auto_manual
    nlg_sys_manual = TemplateNLG(is_user=False, mode='manual')
    nlg_sys_auto = TemplateNLG(is_user=False, mode='auto')
    nlg_sys_auto_manual = TemplateNLG(is_user=False, mode='auto_manual')

    # generate
    print('manual      : ', nlg_sys_manual.generate(dialog_acts))
    print('auto        : ', nlg_sys_auto.generate(dialog_acts))
    print('auto_manual : ', nlg_sys_auto_manual.generate(dialog_acts))

if __name__ == '__main__':
    #test()
    example()
