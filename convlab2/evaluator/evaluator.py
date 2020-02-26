# -*- coding: utf-8 -*-


class Evaluator(object):
    def __init__(self):
        raise NotImplementedError

    def add_goal(self, goal):
        """init goal and array.

        args:
            goal:
                dict[domain] dict['info'/'book'/'reqt'] dict/dict/list[slot]
        """
        raise NotImplementedError

    def add_sys_da(self, da_turn):
        """add sys_da into array.

        args:
            da_turn:
                dict[domain-intent] list[slot, value]
        """
        raise NotImplementedError

    def add_usr_da(self, da_turn):
        """add usr_da into array

        args:
            da_turn:
                dict[domain-intent] list[slot, value]
        """
        raise NotImplementedError

    def book_rate(self, ref2goal=True, aggregate=True):
        """judge if the selected entity meets the constraint"""
        raise NotImplementedError

    def inform_F1(self, ref2goal=True, aggregate=True):
        """judge if all the requested information is answered"""
        raise NotImplementedError

    def task_success(self, ref2goal=True):
        """
        judge if all the domains are successfully completed
        """
        raise NotImplementedError

    def domain_success(self, domain, ref2goal=True):
        """
        judge if the domain (subtask) is successfully completed
        """
        raise NotImplementedError
