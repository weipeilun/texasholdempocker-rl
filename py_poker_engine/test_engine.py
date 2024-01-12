from pypokerengine.api.game import setup_config, start_poker
from py_poker_engine.alpha_go_zero_player import AlphaGoZeroPlayer

config = setup_config(max_round=10, initial_stack=100000, small_blind_amount=25)
config.register_player(name="player_0", algorithm=AlphaGoZeroPlayer())
config.register_player(name="player_1", algorithm=AlphaGoZeroPlayer())
game_result = start_poker(config, verbose=1)
print(game_result)

