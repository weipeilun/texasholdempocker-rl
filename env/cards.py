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
    FIGURE_NAME_DICT = {
        CardFigure.CARD_FIGURE_2: '2',
        CardFigure.CARD_FIGURE_3: '3',
        CardFigure.CARD_FIGURE_4: '4',
        CardFigure.CARD_FIGURE_5: '5',
        CardFigure.CARD_FIGURE_6: '6',
        CardFigure.CARD_FIGURE_7: '7',
        CardFigure.CARD_FIGURE_8: '8',
        CardFigure.CARD_FIGURE_9: '9',
        CardFigure.CARD_FIGURE_10: 'T',
        CardFigure.CARD_FIGURE_JACK: 'J',
        CardFigure.CARD_FIGURE_QUEEN: 'Q',
        CardFigure.CARD_FIGURE_KING: 'K',
        CardFigure.CARD_FIGURE_ACE: 'A',
    }

    DECOR_NAME_DICT = {
        CardDecor.CLUB: 'C',
        CardDecor.DIAMOND: 'D',
        CardDecor.HEART: 'H',
        CardDecor.SPADE: 'S',
    }

    def __init__(self, figure, decor):
        self.figure = figure
        self.decor = decor

    def __cmp__(self, other):
        return cmp(self, other)

    def __str__(self):
        return f'{self.DECOR_NAME_DICT[self.decor]}{self.FIGURE_NAME_DICT[self.figure]}'

    def __hash__(self):
        return hash(self.figure.value * 10 + self.decor.value)
