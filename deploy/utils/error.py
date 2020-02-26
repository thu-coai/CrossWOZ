#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

"""


class DeployError(Exception):
    def __init__(self, msg: str, module: str = 'system', model: str = ''):
        super().__init__(self)
        self.msg = msg
        self.module = module
        self.model = model

    def __str__(self):
        text = ''
        if self.module:
            text += '[%s]' % self.module
        if self.model:
            text += '<%s>' % self.model
        return text + ' ' + self.msg
