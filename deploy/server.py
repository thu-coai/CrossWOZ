#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Backend service response service class
"""
import json
import copy
from deploy.ctrl import ModuleCtrl, SessionCtrl
from deploy.utils import DeployError

MODULES = ['nlu', 'dst', 'policy', 'nlg']


class ServerCtrl(object):
    def __init__(self, **kwargs):
        self.net_conf = copy.deepcopy(kwargs['net'])
        self.module_conf = {
            'nlu': copy.deepcopy(kwargs['nlu']),
            'dst': copy.deepcopy(kwargs['dst']),
            'policy': copy.deepcopy(kwargs['policy']),
            'nlg': copy.deepcopy(kwargs['nlg'])
        }
        self.modules = {mdl: ModuleCtrl(mdl, self.module_conf[mdl]) for mdl in self.module_conf.keys()}
        self.sessions = SessionCtrl(expire_sec=self.net_conf['session_time_out'])

    def on_models(self):
        ret = {}
        for module_name in MODULES:
            ret[module_name] = {}
            for model_id in self.module_conf[module_name].keys():
                ret[module_name][model_id] = {key: self.module_conf[module_name][model_id][key] for key in
                                              ['class_path', 'data_set', 'ini_params', 'model_name']}
                ret[module_name][model_id]['ini_params'] = json.dumps(ret[module_name][model_id]['ini_params'])
        return ret

    def on_register(self, **kwargs):
        ret = {key: 0 for key in MODULES}
        try:
            for module_name in MODULES:
                model_id = kwargs.get(module_name, None)
                if isinstance(model_id, str):
                    ret[module_name] = self.modules[module_name].add_used_num(model_id)
        except Exception as e:
            for module_name in MODULES:
                model_id = kwargs.get(module_name, None)
                if isinstance(model_id, str) and ret[module_name] != 0:
                    self.modules[module_name].sub_used_num(model_id)
            raise e

        if ret['nlu'] == 0 and ret['dst'] == 0 and ret['policy'] == 0 and ret['nlg'] == 0:
            raise DeployError('At least one model needs to be started')

        token = self.sessions.new_session(*[kwargs.get(mn, None) for mn in MODULES])

        return {'token': token}

    def on_close(self, token):
        if not self.sessions.has_token(token):
            raise DeployError('No such token:\'%s\'' % token)

        session = self.sessions.pop_session(token)
        for module in MODULES:
            self.modules[module].sub_used_num(session['model_map'][module])
        return {'del_token': token}

    def on_clear_expire(self):
        expire_session = self.sessions.pop_expire_session()
        del_tokens = []
        for (token, session) in expire_session.items():
            del_tokens.append(token)
            for module in MODULES:
                self.modules[module].sub_used_num(session['model_map'][module])
        return {'del_tokens': del_tokens}

    def on_response(self, token, input_module, data, modified_output={}):
        if not self.sessions.has_token(token):
            raise DeployError('No such token:\'%s\'' % token)
        session = self.sessions.get_session(token)

        cur_turn = self._turn(last_turn=session['turns'][-1] if session['turns'] else None,
                              model_map=session['model_map'],
                              history=ServerCtrl._history_from_turns(session['turns']),
                              input_module=input_module,
                              data=data,
                              modified_output=modified_output)
        session['turns'].append(cur_turn)
        self.sessions.set_session(token, session)

        return ServerCtrl._response_from_session(session['turns'])

    def on_modify_last(self, token, modified_output):
        if not self.sessions.has_token(token):
            raise DeployError('No such token:\'%s\'' % token)
        session = self.sessions.get_session(token)

        if not session['turns']:
            raise DeployError('This is the first turn in this session.')

        last_turn = session['turns'][-1]
        session['turns'] = session['turns'][:-1]

        for (key, value) in modified_output.items():
            last_turn['modified_output'][key] = value

        cur_turn = self._turn(last_turn=session['turns'][-1] if session['turns'] else None,
                              model_map=session['model_map'],
                              history=ServerCtrl._history_from_turns(session['turns']),
                              input_module=last_turn['input_module'],
                              data=last_turn['data'],
                              modified_output=last_turn['modified_output'])
        session['turns'].append(cur_turn)
        self.sessions.set_session(token, session)

        return ServerCtrl._response_from_session(session['turns'])

    def on_rollback(self, token, back_turns=1):
        if not self.sessions.has_token(token):
            raise DeployError('No such token:\'%s\'' % token)
        session = self.sessions.get_session(token)

        session['turns'] = session['turns'][:-back_turns]
        self.sessions.set_session(token, session)

        return ServerCtrl._response_from_session(session['turns'])

    def _turn(self, last_turn, model_map, history, input_module, data, modified_output):
        # params
        modified_output = {mod: modified_output.get(mod, None) for mod in MODULES}
        cur_cache = last_turn['cache'] if last_turn is not None else {name: None for name in MODULES}

        # process
        new_cache = {name: None for name in MODULES}
        model_ret = {name: None for name in MODULES}
        temp_data = None
        for mod in MODULES:
            if input_module == mod:
                temp_data = data

            if temp_data is not None and model_map[mod] is not None:
                (model_ret[mod], new_cache[mod]) = self.modules[mod].run(model_map[mod],
                                                                         cur_cache[mod],
                                                                         last_turn is None,
                                                                         [temp_data, history] if mod == 'nlu' else [temp_data])
                if modified_output[mod] is not None:
                    model_ret[mod] = modified_output[mod]

                temp_data = model_ret[mod]
            elif mod == 'policy':
                temp_data = None

        # save cache
        cur_turn = {
            'data': data, 'input_module': input_module, 'modified_output': modified_output,
            'cache': new_cache,
            'context': {
                'usr': data if isinstance(data, str) and input_module == 'nlu' else '',
                'sys': model_ret['nlg'] if isinstance(model_ret['nlg'], str) else ''
            },
            'return': model_ret
        }

        return cur_turn

    @staticmethod
    def _history_from_turns(turns):
        history = []
        for turn in turns:
            history.append(['user', turn.get('context', {}).get('usr', '')])
            history.append(['system', turn.get('context', {}).get('sys', '')])
            # history.append(turn.get('context', {}).get('usr', ''))
            # history.append(turn.get('context', {}).get('sys', ''))
        return history

    @staticmethod
    def _response_from_session(turns):
        ret = {
            'nlu': turns[-1]['return']['nlu'] if turns else None,
            'dst': turns[-1]['return']['dst'] if turns else None,
            'policy': turns[-1]['return']['policy'] if turns else None,
            'nlg': turns[-1]['return']['nlg'] if turns else None,
            'modified_model': [mod for (mod, val) in turns[-1]['modified_output'].items() if val is not None] if turns else None,
            'history': ServerCtrl._history_from_turns(turns)
        }
        return ret


if __name__ == '__main__':
    pass
