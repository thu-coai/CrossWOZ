#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

"""
import uuid
from deploy.utils import MyLock, ExpireDict


class SessionCtrl(object):

    def __init__(self, max_items=None, expire_sec=None):
        self.sessions = ExpireDict(max_items, expire_sec)
        # self.sessions = ExpireDict(2, expire_sec)
        self.lock = MyLock()

    def get_session(self, token) -> dict:
        return self.sessions[token]

    def set_session(self, token, data):
        self.sessions[token] = data

    def pop_session(self, token) -> dict:
        return self.sessions.pop(token)

    def has_token(self, token) -> bool:
        return token in self.sessions.keys()

    def new_session(self, nlu, dst, policy, nlg) -> str:
        with self.lock:
            token = self.__new_token()
            self.sessions[token] = self.__new_data(nlu, dst, policy, nlg)
        return token

    def pop_expire_session(self):
        return self.sessions.pop_expire()

    def __new_data(self, nlu, dst, policy, nlg):
        return {
            'model_map': {'nlu': nlu, 'dst': dst, 'policy': policy, 'nlg': nlg}, 'turns': []
        }

    def __new_token(self):
        token = str(uuid.uuid4())
        while token in self.sessions.keys():
            token = str(uuid.uuid4())
        return token


if __name__ == '__main__':
    pass
