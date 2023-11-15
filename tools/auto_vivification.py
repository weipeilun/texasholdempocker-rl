#!/usr/bin/env python
# encoding: utf-8

"""
Author:weipeilun
E-mail:weipeilun0217@gmail.com
"""


class AutoVivification(dict):
    """Implementation of perl's autovivification feature."""
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value
