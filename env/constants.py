from enum import Enum

MAX_PLAYER_NUMBER = 2

# card figures
class CardFigure(Enum):
    CARD_FIGURE_2 = 0
    CARD_FIGURE_3 = 1
    CARD_FIGURE_4 = 2
    CARD_FIGURE_5 = 3
    CARD_FIGURE_6 = 4
    CARD_FIGURE_7 = 5
    CARD_FIGURE_8 = 6
    CARD_FIGURE_9 = 7
    CARD_FIGURE_10 = 8
    CARD_FIGURE_JACK = 9
    CARD_FIGURE_QUEUE = 10
    CARD_FIGURE_KING = 11
    CARD_FIGURE_ACE = 12

class CardDecor(Enum):
    HEART = 0
    SPADE = 1
    CLUB = 2
    DIAMOND = 3

# betting round action
class PlayerActions(Enum):
    FOLD = 0
    CHECK_CALL = 1
    RAISE = 2
    BIG_BLIND_RAISE = 3
    SMALL_BLIND_RAISE = 4

# players status
class PlayerStatus(Enum):
    ONBOARD = 0
    ALLIN = 1
    FOLDED = 2
    BUSTED = 3

# players status
class CardCombinations(Enum):
    ONES = 0
    PAIRS = 1
    TRIPLE = 2
    QUATRE = 3
    STRAIGHTS = 4
    FLUSH = 5


# game player result
class GamePlayerResult(Enum):
    DEFAULT = None
    LOSE = -1.
    EVEN = 0.
    WIN = 1.


class WorkflowStatus(Enum):
    TRAINING = 0
    TRAIN_FINISH_WAIT = 1
    REGISTERING_EVAL_MODEL = 2
    EVALUATING = 3
    EVAL_FINISH_WAIT = 4
    REGISTERING_TRAIN_MODEL = 5
    DEFAULT = 0
    UNDEFINED = -1


class ChoiceMethod(Enum):
    ARGMAX = 'argmax'
    PROBABILITY = 'probability'

# model action bin dict
ACTION_BINS_DICT = [
    (PlayerActions.FOLD, (0, 0)),
    (PlayerActions.CHECK_CALL, (0, 0)),
    (PlayerActions.RAISE, (0., 0.0125)),
    (PlayerActions.RAISE, (0.0125, 0.025)),
    (PlayerActions.RAISE, (0.025, 0.05)),
    (PlayerActions.RAISE, (0.05, 0.1)),
    (PlayerActions.RAISE, (0.1, 0.2)),
    (PlayerActions.RAISE, (0.2, 0.4)),
    (PlayerActions.RAISE, (0.4, 0.6)),
    (PlayerActions.RAISE, (0.6, 0.8)),
    (PlayerActions.RAISE, (0.8, 1.)),
    (PlayerActions.RAISE, (1., 1.)),
]


NUM_CARDS = len(CardFigure) * len(CardDecor)

# players
_PLAYER_PREFIX = 'player_'
GET_PLAYER_NAME = lambda key: f'{_PLAYER_PREFIX}{key}'
GET_PLAYER_ID_BY_NAME = lambda name: int(name.lstrip(_PLAYER_PREFIX))

# envs
CARDS_FLOP = 'cards_flop'
CARDS_TURN = 'cards_turn'
CARDS_RIVER = 'cards_river'

KEY_ACTED_PLAYER_NAME = 'acted_player_name'
KEY_ROUND_NUM = 'round_num'
STEP_ID_FINISHED = -1
