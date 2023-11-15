from env.constants import *
import sys


PY3 = sys.version_info[0] >= 3
if PY3:
    def cmp(a, b):
        if a.figure.value > b.figure.value:
            return 1
        elif a.figure.value < b.figure.value:
            return -1
        else:
            if a.decor.value > b.decor.value:
                return 1
            elif a.decor.value < b.decor.value:
                return -1
            else:
                return 0
    # mixin class for Python3 supporting __cmp__
    class PY3__cmp__:
        def __eq__(self, other):
            return self.__cmp__(other) == 0
        def __ne__(self, other):
            return self.__cmp__(other) != 0
        def __gt__(self, other):
            return self.__cmp__(other) > 0
        def __lt__(self, other):
            return self.__cmp__(other) < 0
        def __ge__(self, other):
            return self.__cmp__(other) >= 0
        def __le__(self, other):
            return self.__cmp__(other) <= 0
else:
    class PY3__cmp__:
        pass


class Card(PY3__cmp__):

    def __init__(self, figure, decor):
        self.figure = figure
        self.decor = decor

    def __cmp__(self, other):
        return cmp(self, other)

    def __str__(self):
        return f'figure:{self.figure}, decor:{self.decor.name}'

    def __hash__(self):
        return hash(self.figure.value * 10 + self.decor.value)
