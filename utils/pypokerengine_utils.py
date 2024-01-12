from env.pypokerengine_constants import *
from env.cards import *


def map_pypokerengine_card_to_env_card(card):
    decor = CARD_DECOR_MAP[card[0]]
    figure = CARD_FIGURE_MAP[card[1]]
    return Card(figure, decor)

