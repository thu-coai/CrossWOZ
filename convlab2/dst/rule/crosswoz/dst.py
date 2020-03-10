from convlab2.dst.dst import DST
from convlab2.util.crosswoz.state import default_state
from convlab2.util.crosswoz.dbquery import Database
from copy import deepcopy
from collections import Counter
from pprint import pprint


class RuleDST(DST):
    """Rule based DST which trivially updates new values from NLU result to states.

    Attributes:
        state(dict):
            Dialog state. Function ``convlab2.util.crosswoz.state.default_state`` returns a default state.
    """
    def __init__(self):
        super().__init__()
        self.state = default_state()
        self.database = Database()

    def init_session(self, state=None):
        """Initialize ``self.state`` with a default state, which ``convlab2.util.crosswoz.state.default_state`` returns."""
        self.state = default_state() if not state else deepcopy(state)

    def update(self, usr_da=None):
        """
        update belief_state, cur_domain, request_slot
        :param usr_da:
        :return:
        """
        sys_da = self.state['system_action']

        select_domains = Counter([x[1] for x in usr_da if x[0] == 'Select'])
        request_domains = Counter([x[1] for x in usr_da if x[0] == 'Request'])
        inform_domains = Counter([x[1] for x in usr_da if x[0] == 'Inform'])
        sys_domains = Counter([x[1] for x in sys_da if x[0] in ['Inform', 'Recommend']])
        if len(select_domains) > 0:
            self.state['cur_domain'] = select_domains.most_common(1)[0][0]
        elif len(request_domains) > 0:
            self.state['cur_domain'] = request_domains.most_common(1)[0][0]
        elif len(inform_domains) > 0:
            self.state['cur_domain'] = inform_domains.most_common(1)[0][0]
        elif len(sys_domains) > 0:
            self.state['cur_domain'] = sys_domains.most_common(1)[0][0]
        else:
            self.state['cur_domain'] = None

        # print('cur_domain', self.cur_domain)

        NoOffer = 'NoOffer' in [x[0] for x in sys_da] and 'Inform' not in [x[0] for x in sys_da]
        # DONE: clean cur domain constraints because nooffer

        if NoOffer:
            if self.state['cur_domain']:
                self.state['belief_state'][self.state['cur_domain']] = deepcopy(default_state()['belief_state'][self.state['cur_domain']])

        # DONE: clean request slot
        for domain, slot in deepcopy(self.state['request_slots']):
            if [domain, slot] in [x[1:3] for x in sys_da if x[0] in ['Inform', 'Recommend']]:
                self.state['request_slots'].remove([domain, slot])

        # DONE: domain switch
        for intent, domain, slot, value in usr_da:
            if intent == 'Select':
                from_domain = value
                name = self.state['belief_state'][from_domain]['名称']
                if name:
                    if domain == from_domain:
                        self.state['belief_state'][domain] = deepcopy(default_state()['belief_state'][domain])
                    self.state['belief_state'][domain]['周边{}'.format(from_domain)] = name

        for intent, domain, slot, value in usr_da:
            if intent == 'Inform':
                if slot in ['名称', '游玩时间', '酒店类型', '出发地', '目的地', '评分', '门票', '价格', '人均消费']:
                    self.state['belief_state'][domain][slot] = value
                elif slot == '推荐菜':
                    if not self.state['belief_state'][domain][slot]:
                        self.state['belief_state'][domain][slot] = value
                    else:
                        self.state['belief_state'][domain][slot] += ' ' + value
                elif '酒店设施' in slot:
                    if value == '是':
                        faci = slot.split('-')[1]
                        if not self.state['belief_state'][domain]['酒店设施']:
                            self.state['belief_state'][domain]['酒店设施'] = faci
                        else:
                            self.state['belief_state'][domain]['酒店设施'] += ' ' + faci
            elif intent == 'Request':
                self.state['request_slots'].append([domain, slot])

        return self.state

    def query(self):
        return self.database.query(self.state['belief_state'], self.state['cur_domain'])


if __name__ == '__main__':
    dst = RuleDST()
    dst.init_session()
    pprint(dst.state)
    dst.update([['Inform', '酒店', '评分', '4分以上'],['Request', '酒店', '地址', '']])
    pprint(dst.state)
    # pprint(dst.query())
