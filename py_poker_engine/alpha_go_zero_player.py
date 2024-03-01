import logging

from pypokerengine.players import BasePokerPlayer
from env.pypokerengine_env import PyPokerEngineEnv
from tools.param_parser import *
from utils.pypokerengine_utils import *
from rl.MCTS import SingleThreadMCTS
from rl.AlphaGoZero import AlphaGoZero
import torch
import os


class AlphaGoZeroPlayer(BasePokerPlayer):

    def __init__(self):
        super().__init__()

        # MCTS依赖ENV做模拟，在此必须初始化env并在每一步中做状态校验
        self.env = None
        self.player_name = None
        self.player_id = None
        self.params = parse_params()[1]
        # 相同的player在declare_action之后会马上收到一个receive_game_update_message，将其忽略
        self.has_just_taken_action = False

        self.hand_cards = None
        self.uuid_player_name_dict = None

        self.observation = None

        model_param_dict = self.params['model_param_dict']
        self.model = AlphaGoZero(**model_param_dict)
        model_path = self.params['model_init_checkpoint_path']
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            model_param = checkpoint['model']
            self.model.load_state_dict(model_param, strict=False)
            if 'optimizer' in checkpoint:
                self.model.optimizer.load_state_dict(checkpoint['optimizer'])
            logging.info(f'model loaded from {model_path}')
        else:
            logging.warning(f'model not found at {model_path}')

        self.num_output_class = model_param_dict['num_output_class']
        self.historical_action_per_round = model_param_dict['historical_action_per_round']
        self.small_blind = self.params['small_blind']
        self.num_mcts_simulation_per_step = self.params['num_mcts_simulation_per_step']
        self.mcts_c_puct = self.params['mcts_c_puct']
        self.mcts_model_Q_epsilon = self.params['mcts_model_Q_epsilon']
        self.mcts_log_to_file = self.params['mcts_log_to_file']
        self.mcts_choice_method = self.params['mcts_choice_method']

    #  we define the logic to make an action through this method. (so this method would be the core of your AI)
    def declare_action(self, valid_actions, hole_card, round_state):
        mcts = SingleThreadMCTS(self.num_output_class,
                                is_root=True,
                                apply_dirichlet_noice=False,
                                small_blind=self.small_blind,
                                model=self.model,
                                n_simulation=self.num_mcts_simulation_per_step,
                                c_puct=self.mcts_c_puct,
                                tau=0,
                                model_Q_epsilon=self.mcts_model_Q_epsilon,
                                choice_method=self.mcts_choice_method,
                                log_to_file=self.mcts_log_to_file,
                                pid=self.player_id)
        action_probs = mcts.simulate(observation=self.observation, env=self.env)
        # select action by probability in play is a special need by poker to distract opponents, which is different from chess and go
        action_bin, action, action_mask_idx = mcts.get_action(action_probs, env=self.env, choice_method=ChoiceMethod.PROBABILITY)

        observation, _, terminated, _ = self.env.step(action, action_bin)
        if not terminated:
            self.observation = observation
        self.has_just_taken_action = True
        return ACTION_TO_ACTION_REVERSE_MAP[action[0]], action[1]   # action returned here is sent to the poker engine

    def receive_game_start_message(self, game_info):
        self.uuid_player_name_dict = dict()
        for seat in game_info['seats']:
            if seat['uuid'] == self.uuid:
                self.player_name = seat['name']
                self.player_id = GET_PLAYER_ID_BY_NAME(self.player_name)
            self.uuid_player_name_dict[seat['uuid']] = seat['name']

        if self.env is None:
            self.env = PyPokerEngineEnv(None, self.num_output_class, historical_action_per_round=self.historical_action_per_round, num_players=MAX_PLAYER_NUMBER, small_blind=game_info['rule']['small_blind_amount'], big_blind=game_info['rule']['small_blind_amount'] * 2, init_value=game_info['rule']['initial_stack'], ignore_all_async_tasks=True)
        self.env.check_game_info(game_info, self.params)

    def receive_round_start_message(self, round_count, hole_card, seats):
        self.hand_cards = sorted(map_pypokerengine_card_to_env_card(card) for card in hole_card)
        self.observation = self.env.reset(game_id=round_count, cards_dict={self.player_name: self.hand_cards})
        self.env.check_seats(seats)
        self.env.check_player_cards(self.player_name, self.hand_cards)

    def receive_street_start_message(self, street, round_state):
        is_all_player_all_in = True
        for player in round_state['seats']:
            if player['state'] != 'allin':
                is_all_player_all_in = False
                break

        if not is_all_player_all_in:
            cards_dict = dict()
            cards_dict[self.player_name] = self.hand_cards.copy()
            if STREET_TO_CURRENT_ROUND_MAP[street] >= 1:
                cards_dict[CARDS_FLOP] = sorted(map_pypokerengine_card_to_env_card(card) for card in round_state['community_card'][:3])
            if STREET_TO_CURRENT_ROUND_MAP[street] >= 2:
                cards_dict[CARDS_TURN] = sorted(map_pypokerengine_card_to_env_card(card) for card in round_state['community_card'][3:4])
            if STREET_TO_CURRENT_ROUND_MAP[street] >= 3:
                cards_dict[CARDS_RIVER] = sorted(map_pypokerengine_card_to_env_card(card) for card in round_state['community_card'][4:5])
            self.env.reinit_cards(cards_dict=cards_dict)

            self.env.check_street_state(round_state, self.uuid_player_name_dict)

    def receive_game_update_message(self, action, round_state):
        if self.has_just_taken_action:
            self.has_just_taken_action = False
        else:
            self.env.check_others_action(action, self.uuid_player_name_dict)

            mapped_action = (ACTION_TO_ACTION_MAP[action['action']], action['amount'])
            # todo：此处特征拼接有问题：action_bin为空，在message_builder.py:63
            self.env.step(mapped_action, None)
            self.env.check_round_state(round_state, self.uuid_player_name_dict)

    def receive_round_result_message(self, winners, hand_info, round_state):
        cards_dict = dict()
        for hand in hand_info:
            cards_dict[self.uuid_player_name_dict[hand['uuid']]] = sorted(map_pypokerengine_card_to_env_card(card) for card in hand['hole_card'])
        cards_dict[CARDS_FLOP] = sorted(map_pypokerengine_card_to_env_card(card) for card in round_state['community_card'][:3])
        cards_dict[CARDS_TURN] = sorted(map_pypokerengine_card_to_env_card(card) for card in round_state['community_card'][3:4])
        cards_dict[CARDS_RIVER] = sorted(map_pypokerengine_card_to_env_card(card) for card in round_state['community_card'][4:5])
        self.env.reinit_cards(cards_dict=cards_dict)
        winner_value_dict = self.env.settle_game()
        # todo: check hand result
        self.env.check_round_result(winner_value_dict, winners, round_state)


def setup_ai():
    return AlphaGoZeroPlayer()
