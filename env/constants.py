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
    CHECK = 0
    RAISE = 1
    CALL = 2
    FOLD = 3
    BIG_BLIND_RAISE = 4
    SMALL_BLIND_RAISE = 5
    NOT_AN_ACTION = 6

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
