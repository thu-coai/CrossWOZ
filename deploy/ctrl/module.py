#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Control the invocation of all modules
"""
import copy
from deploy.ctrl.model import ModelCtrl
from deploy.utils import DeployError


class ModuleCtrl(object):
    mod2method = {'nlu': 'predict', 'dst': 'update', 'policy': 'predict', 'nlg': 'generate'}

    def __init__(self, module_name: str, infos: dict):
        assert module_name in self.mod2method.keys(), 'Unknow module name \'%s\'' % module_name
        self.module_name = module_name
        self.method = self.mod2method[self.module_name]
        self.infos = copy.deepcopy(infos)
        self.models = {mid: ModelCtrl(mid, **self.infos[mid]) for mid in self.infos.keys()}

    def add_used_num(self, model_id: str):
        try:
            self.models[model_id].add_used_num()
        except TypeError:
            raise DeployError('Unknow model id \'%s\'' % model_id, module=self.module_name)

    def sub_used_num(self, model_id: str):
        if model_id is not None:
            try:
                self.models[model_id].sub_used_num()
            except TypeError:
                raise DeployError('Unknow model id \'%s\'' % model_id, module=self.module_name)

    def run(self, model_id, cache, isfirst, params):
        try:
            ret = self.models[model_id].run(self.method, cache, isfirst, params)
        except TypeError:
            raise DeployError('Unknow model id \'%s\'' % model_id, module=self.module_name)
        return ret


if __name__ == '__main__':
    from deploy.config import get_config

    conf = get_config()
    aaa = ModuleCtrl(**conf)
