#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

"""
import copy
from deploy.utils import MyLock, ResourceLock, DeployError


class ModelCtrl(object):
    def __init__(self, *args, **kwargs):
        # model id
        self.model_id = args[0]

        # running params
        self.model_class = kwargs['class']
        self.ini_params = kwargs.get('ini_params', dict({}))
        self.max_core = kwargs.get('max_core', 1)

        # do not care
        self.class_path = kwargs.get('class_path', '')
        self.model_name = kwargs.get('model_name', '')
        self.data_set = kwargs.get('data_set', '')

        self.opt_lock = MyLock()
        self.used_num = 0

        self.models = [None for _ in range(self.max_core)]
        self.res_lock = ResourceLock(self.max_core)

        if kwargs.get('preload', False):
            print('Model [%s] Preload' % self.model_id)
            self.add_used_num()

    def add_used_num(self):
        with self.opt_lock:
            if self.used_num == 0:
                print('------------implement: ' + self.model_id)
                res_idxs = self.__catch_all_res()
                try:
                    self.models = [self.__implement() for _ in range(self.max_core)]
                except Exception as e:
                    raise DeployError('Instantiation failed:%s' % str(e), model=self.model_id)
                finally:
                    self.__leave_all_res(res_idxs)
            self.used_num += 1
        return self.used_num

    def sub_used_num(self):
        with self.opt_lock:
            self.used_num -= 1
            self.used_num = 0 if self.used_num < 0 else self.used_num
            if self.used_num == 0:
                print('------------destroy: ' + self.model_id)
                res_idxs = self.__catch_all_res()
                for mod in self.models:
                    if mod is not None:
                        del mod
                self.models = [None for _ in range(self.max_core)]
                self.__leave_all_res(res_idxs)
        return self.used_num

    def run(self, method, cache, isfirst, params):
        res_idx = self.res_lock.res_catch()
        print('+++++ catch res ' + str(res_idx) + ' ' + self.model_id)
        try:
            # get model
            model = self.models[res_idx]
            if model is None:
                raise DeployError('Model has not started yet.', model=self.model_id)

            # load cache
            if isfirst:
                getattr(model, 'init_session')()  # first turn
            else:
                getattr(model, 'from_cache')(cache)

            # process
            ret_data = getattr(model, method)(*params)

            # save cache
            new_cache = copy.deepcopy(getattr(model, 'to_cache')())

        except Exception as e:
            if not isinstance(e, DeployError):
                raise DeployError('running error:%s' % str(e), model=self.model_id)
            else:
                raise e
        finally:
            print('----- leave res ' + str(res_idx) + ' ' + self.model_id)
            self.res_lock.res_leave(res_idx)

        return ret_data, new_cache

    def __catch_all_res(self):
        res_idxs = []
        for _ in range(self.max_core):
            res_idxs.append(self.res_lock.res_catch())
        return res_idxs

    def __leave_all_res(self, res_idxs):
        for idx in res_idxs[::-1]:
            self.res_lock.res_leave(idx)

    def __implement(self):
        return self.model_class(**self.ini_params)


if __name__ == '__main__':
    pass
