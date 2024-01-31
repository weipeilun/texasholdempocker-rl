from pypokerengine.api.game import setup_config, start_poker
from py_poker_engine.fish_player import FishPlayer
# from py_poker_engine.alpha_go_zero_player import AlphaGoZeroPlayer
import logging

log_level = logging.INFO
logging.basicConfig(format='%(asctime)s.%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d,%H:%M:%S')
logger = logging.getLogger()
logger.setLevel(log_level)

config = setup_config(max_round=10, initial_stack=100000, small_blind_amount=25)
config.register_player(name="player_0", algorithm=FishPlayer())
config.register_player(name="player_1", algorithm=FishPlayer())
game_result = start_poker(config, verbose=1)
print(game_result)

