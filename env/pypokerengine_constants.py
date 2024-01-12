from .constants import *
from pypokerengine.engine.data_encoder import DataEncoder
from pypokerengine.engine.player import Player


# pypokerengine.engine.card.py:8
CARD_DECOR_MAP = {
    'H': CardDecor.HEART,
    'S': CardDecor.SPADE,
    'C': CardDecor.CLUB,
    'D': CardDecor.DIAMOND,
}

# pypokerengine.engine.card.py:15
CARD_FIGURE_MAP = {
    '2': CardFigure.CARD_FIGURE_2,
    '3': CardFigure.CARD_FIGURE_3,
    '4': CardFigure.CARD_FIGURE_4,
    '5': CardFigure.CARD_FIGURE_5,
    '6': CardFigure.CARD_FIGURE_6,
    '7': CardFigure.CARD_FIGURE_7,
    '8': CardFigure.CARD_FIGURE_8,
    '9': CardFigure.CARD_FIGURE_9,
    'T': CardFigure.CARD_FIGURE_10,
    'J': CardFigure.CARD_FIGURE_JACK,
    'Q': CardFigure.CARD_FIGURE_QUEUE,
    'K': CardFigure.CARD_FIGURE_KING,
    'A': CardFigure.CARD_FIGURE_ACE
}

PLAYER_STATUS_MAP = {
    DataEncoder.PAY_INFO_PAY_TILL_END_STR: PlayerStatus.ONBOARD,
    DataEncoder.PAY_INFO_ALLIN_STR: PlayerStatus.ALLIN,
    DataEncoder.PAY_INFO_FOLDED_STR: PlayerStatus.FOLDED,
}

# pypokerengine.engine.data_encoder.py:123
STREET_PREFLOP = 'preflop'
STREET_FLOP = 'flop'
STREET_TURN = 'turn'
STREET_RIVER = 'river'
STREET_SHOWDOWN = 'showdown'
STREET_TO_CURRENT_ROUND_MAP = {
    STREET_PREFLOP: 0,
    STREET_FLOP: 1,
    STREET_TURN: 2,
    STREET_RIVER: 3,
    STREET_SHOWDOWN: 4,
}

# pypokerengine.utils.action_utils.py:3
ACTION_FOLD = 'fold'
ACTION_CALL = 'call'
ACTION_RAISE = 'raise'
ACTION_TO_ACTION_MAP = {
    ACTION_FOLD: PlayerActions.FOLD,
    ACTION_CALL: PlayerActions.CHECK_CALL,
    ACTION_RAISE: PlayerActions.RAISE,
}
ACTION_TO_ACTION_REVERSE_MAP = {v: k for k, v in ACTION_TO_ACTION_MAP.items()}

ACTION_PLAYER_TO_ACTION_MAP = {
    Player.ACTION_FOLD_STR: PlayerActions.FOLD,
    Player.ACTION_CALL_STR: PlayerActions.CHECK_CALL,
    Player.ACTION_RAISE_STR: PlayerActions.RAISE,
    Player.ACTION_SMALL_BLIND: PlayerActions.SMALL_BLIND_RAISE,
    Player.ACTION_BIG_BLIND: PlayerActions.BIG_BLIND_RAISE,
}