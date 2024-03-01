from .env import Env
from utils.format_utils import *
from utils.pypokerengine_utils import *
from pypokerengine.engine.player import Player


class PyPokerEngineEnv(Env):

    def __init__(self, winning_probability_generating_task_queue, num_action_bins, historical_action_per_round, num_players: int = MAX_PLAYER_NUMBER, init_value: int = 100_000, small_blind=25, big_blind=50, num_player_fields=3, game_env=None, ignore_all_async_tasks=False, settle_automatically=False):
        super().__init__(winning_probability_generating_task_queue=winning_probability_generating_task_queue,
                         num_action_bins=num_action_bins,
                         historical_action_per_round=historical_action_per_round,
                         num_players=num_players,
                         init_value=init_value,
                         small_blind=small_blind,
                         big_blind=big_blind,
                         num_player_fields=num_player_fields,
                         game_env=game_env,
                         ignore_all_async_tasks=ignore_all_async_tasks,
                         settle_automatically=settle_automatically
                         )

    def reinit_cards(self, seed=None, cards_dict=None):
        return self._env.card_play_init(seed=seed, cards_dict=cards_dict)

    def check_game_info(self, game_info, params):
        assert self.num_players == game_info['player_num'], ValueError(f"num_players:{self.num_players} not equals game_info:{game_info['player_num']}")
        assert self.init_value == game_info['rule']['initial_stack'], ValueError(f"init_value:{self.init_value} not equals game_info:{game_info['rule']['initial_stack']}")
        assert self.small_blind == game_info['rule']['small_blind_amount'], ValueError(f"small_blind:{self.small_blind} not equals game_info:{game_info['rule']['small_blind_amount']}")
        assert self.big_blind == game_info['rule']['small_blind_amount'] * 2, ValueError(f"big_blind:{self.big_blind} not equals game_info:{game_info['rule']['small_blind_amount'] * 2}")

        assert self.small_blind == params['small_blind'], ValueError(f"small_blind:{self.small_blind} not equals params:{params['small_blind']}")
        assert self.big_blind == params['big_blind'], ValueError(f"big_blind:{self.big_blind} not equals params:{params['big_blind']}")

    def check_seats(self, seats):
        assert len(self._env.players) == len(seats), ValueError(f"num_players:{self._env.players} not equals game_info:{len(seats)}")
        for player_name, agent in self._env.players.items():
            player_id = GET_PLAYER_ID_BY_NAME(player_name)
            player_seat = seats[player_id]

            assert player_name == player_seat['name'], ValueError(f"player_name:{player_name} not equals to seat:{player_seat['name']}")
            assert agent.value_left == player_seat['stack'], ValueError(f"player:{player_name} value_left:{agent.value_left} not equals to seat:{player_seat['stack']}")
            seat_status = PLAYER_STATUS_MAP[player_seat['state']]
            if seat_status == PlayerStatus.ALLIN and player_seat['stack'] == 0:
                seat_status = PlayerStatus.BUSTED
            assert agent.status == seat_status, ValueError(f"player:{player_name} status:{agent.status} not equals to seat:{player_seat['state']}")

    def check_player_cards(self, player_name, cards):
        assert self._env.info_sets[player_name].player_hand_cards == cards, ValueError(f"player:{player_name} cards:{FORMAT_CARDS(self._env.info_sets[player_name].player_hand_cards)} not equals to seat:{[str(card) for card in cards]}")

    def check_street_state(self, round_state, uuid_player_name_dict):
        assert self._env.current_round == STREET_TO_CURRENT_ROUND_MAP[round_state['street']], ValueError(f"current_round:{self._env.current_round} not equals round_state:{round_state['street']}")

        converted_cards = [map_pypokerengine_card_to_env_card(card) for card in round_state['community_card']]
        if self._env.current_round == 0:
            assert len(converted_cards) == 0, ValueError(f"wrong card number:{len(converted_cards)} in round 0")
        elif self._env.current_round == 1:
            assert len(converted_cards) == 3, ValueError(f"wrong card number:{len(converted_cards)} in round 1")
            for idx in [0, 1, 2]:
                converted_card = converted_cards[idx]
                assert converted_card in self._env.flop_cards, ValueError(f"card {round_state['community_card']['idx']} not in flop_cards:{FORMAT_CARDS(self._env.flop_cards)}")
        elif self._env.current_round == 2:
            assert len(converted_cards) == 4, ValueError(f"wrong card number:{len(converted_cards)} in round 2")
            for idx in [0, 1, 2]:
                converted_card = converted_cards[idx]
                assert converted_card in self._env.flop_cards, ValueError(f"card {round_state['community_card'][idx]} not in flop_cards:{FORMAT_CARDS(self._env.flop_cards)}")
            converted_card = converted_cards[3]
            assert converted_card in self._env.turn_cards, ValueError(f"card {round_state['community_card'][idx]} not in turn_cards:{FORMAT_CARDS(self._env.turn_cards)}")
        elif self._env.current_round == 3:
            assert len(converted_cards) == 5, ValueError(f"wrong card number:{len(converted_cards)} in round 3")
            for idx in [0, 1, 2]:
                converted_card = converted_cards[idx]
                assert converted_card in self._env.flop_cards, ValueError(f"card {round_state['community_card'][idx]} not in flop_cards:{FORMAT_CARDS(self._env.flop_cards)}")
            converted_card = converted_cards[3]
            assert converted_card in self._env.turn_cards, ValueError(f"card {round_state['community_card'][3]} not in turn_cards:{FORMAT_CARDS(self._env.turn_cards)}")
            converted_card = converted_cards[4]
            assert converted_card in self._env.river_cards, ValueError(f"card {round_state['community_card'][4]} not in river_cards:{FORMAT_CARDS(self._env.river_cards)}")

        assert self._env.button_player_name == GET_PLAYER_NAME(round_state['dealer_btn']), ValueError(f"button_player_name:{self._env.button_player_name} not equals round_state:{round_state['dealer_btn']}")
        assert self._env.acting_player_name == GET_PLAYER_NAME(round_state['next_player']), ValueError(f"acting_player_name:{self._env.acting_player_name} not equals round_state:{round_state['next_player']}")
        assert self.game_id == round_state['round_count'], ValueError(f"game_id:{self._env.game_id} not equals round_state:{round_state['round_count']}")
        assert self.small_blind == round_state['small_blind_amount'], ValueError(f"game_id:{self.small_blind} not equals round_state:{round_state['small_blind_amount']}")
        assert self.big_blind == round_state['small_blind_amount'] * 2, ValueError(f"game_id:{self.big_blind} not equals round_state:{round_state['small_blind_amount'] * 2}")

        self.check_round_state(round_state, uuid_player_name_dict)

    def check_history(self, action_histories, uuid_player_name_dict):
        # to prevent empty action_histories
        modified_action_histories = dict()
        for street_name, action_list in action_histories.items():
            if len(action_list) > 0:
                modified_action_histories[street_name] = action_list

        assert len(self._env.all_round_player_action_value_dict) == len(modified_action_histories), ValueError(f"length of round does not match: all_round_player_action_value_dict:{self._env.all_round_player_action_value_dict}, action_histories:{len(modified_action_histories)}")
        for street_name, action_list in modified_action_histories.items():
            assert STREET_TO_CURRENT_ROUND_MAP[street_name] in self._env.all_round_player_action_value_dict, ValueError(f"street:{street_name} not in all_round_player_action_value_dict:{list(modified_action_histories.keys())}")
            round_player_action_value = self._env.all_round_player_action_value_dict[STREET_TO_CURRENT_ROUND_MAP[street_name]]

            assert len(action_list) == len(round_player_action_value), ValueError(f"length of actions round does not match in street {street_name}: action_list:{len(action_list)}, round_player_action_value:{len(round_player_action_value)}")
            for round_player_action, action in zip(round_player_action_value, action_list):
                assert round_player_action[0] == uuid_player_name_dict[action['uuid']], ValueError(f"historical action player_name:{round_player_action[0]} not equals to action:{uuid_player_name_dict[action['uuid']]}")
                assert round_player_action[1] == ACTION_PLAYER_TO_ACTION_MAP[action['action']], ValueError(f"historical action:{round_player_action[1]} not equals to action:{action['action']}")
                if action['action'] != Player.ACTION_FOLD_STR:
                    assert round_player_action[2] == action['amount'], ValueError(f"historical action value:{round_player_action[2]} not equals to action:{action['amount']}")
                    action_delta_value = action['add_amount'] if round_player_action[1] in (PlayerActions.SMALL_BLIND_RAISE, PlayerActions.BIG_BLIND_RAISE) else action['paid']
                    assert round_player_action[3] == action_delta_value, ValueError(f"historical action delta value:{round_player_action[3]} not equals to action:{action_delta_value}")

    # round_state的很多字段在中间状态中更新不及时，大部分只在check_street_state中检查
    def check_round_state(self, round_state, uuid_player_name_dict):
        # todo：目前只支持两个玩家，后续需要区分
        assert self._env.pot_value == round_state['pot']['main']['amount'], ValueError(f"all_round_overall_value:{self._env.all_round_overall_value} not equals round_state:{round_state['pot']['main']['amount']}")

        self.check_seats(round_state['seats'])
        self.check_history(round_state['action_histories'], uuid_player_name_dict)

    def check_round_result(self, winner_value_dict, winners, round_state):
        assert len(winner_value_dict) == len(winners), ValueError(f"length of env winner:{winner_value_dict} not equals to winners:{winners}")
        for winner in winners:
            assert winner['name'] in winner_value_dict, ValueError(f"winner:{winner} not in env winner:{winner_value_dict}")

        self.check_seats(round_state['seats'])

    def check_others_action(self, action, uuid_player_name_dict):
        assert self._env.acting_player_name == uuid_player_name_dict[action['player_uuid']], ValueError(f"acting_player_name {self._env.acting_player_name} not equals to other_player_name:{uuid_player_name_dict[action['player_uuid']]}, uuid:{action['player_uuid']}")

    def settle_game(self):
        assert self._env.game_over, ValueError(f"env is not game_over when pypokerengine round is finished")
        assert not self.settle_automatically, ValueError(f"can not call settle_game() if self.settle_automatically is True")
        return self._env.finish_game()
