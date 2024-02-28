import logging
import copy
import random
import numpy as np
from .cards import *
from .agents import DummyAgent
from copy import deepcopy
from operator import itemgetter
from math import floor
from phevaluator import evaluate_cards
from utils.pokerhandevaluator_utils import *


deck = []
for figure in CardFigure:
    for decor in CardDecor:
        deck.append(Card(figure, decor))
deck_reverse_dict = {card: idx for idx, card in enumerate(deck)}


# 注意GameEnv中所有value的物理含义都是：当前轮中/当前行动中下注到某个特定数额，不是增量下注数额
class GameEnv(object):

    COMPARE_RULES = [
            [CardCombinations.STRAIGHTS, CardCombinations.FLUSH],
            [CardCombinations.QUATRE],
            [CardCombinations.TRIPLE, CardCombinations.PAIRS],
            [CardCombinations.FLUSH],
            [CardCombinations.STRAIGHTS],
            [CardCombinations.TRIPLE, CardCombinations.ONES],
            [CardCombinations.PAIRS, CardCombinations.ONES],
            [CardCombinations.ONES],
        ]

    def __init__(self, num_players, init_value=100_000, small_blind=25, big_blind=50, settle_automatically=True):
        self.initial_num_players = num_players
        self.initial_init_value = init_value
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.settle_automatically = settle_automatically

        self.num_players = self.initial_num_players
        self.init_value = self.initial_init_value

        self.players = None

        # king_player_name实际上是button位
        self.button_player_name = None
        self.acting_player_name = None

        self.player_init_value_dict = None

        self.flop_cards = None
        self.turn_cards = None
        self.river_cards = None

        self.game_over = False
        self.winner = None

        self.current_round = 0
        self.current_round_min_value = 0
        self.current_round_raise_min_value = 0
        self.current_round_action_dict = None
        self.current_round_value_dict = None
        self.historical_round_action_list = None
        self.historical_round_value_list = None

        # current_round -> player_name, action, bet, delta_bet
        self.all_round_player_action_value_dict = None

        self.info_sets = None
        self.game_infoset = None

        # 总下注金额
        self.pot_value = 0

        self.ignore_all_async_tasks = False

    def new_random(self):
        # new_random只会在MCTS模拟中用到，所以设置settle_automatically为True
        new_env = GameEnv(self.initial_num_players, self.initial_init_value, self.small_blind, self.big_blind, settle_automatically=True)

        new_env.players = copy.deepcopy(self.players)

        new_env.button_player_name = self.button_player_name
        new_env.acting_player_name = self.acting_player_name

        new_env.player_init_value_dict = copy.deepcopy(self.player_init_value_dict)

        new_env.flop_cards = copy.deepcopy(self.flop_cards)
        new_env.turn_cards = copy.deepcopy(self.turn_cards)
        new_env.river_cards = copy.deepcopy(self.river_cards)

        new_env.game_over = self.game_over
        new_env.winner = self.winner

        new_env.current_round = self.current_round
        new_env.current_round_min_value = self.current_round_min_value
        new_env.current_round_raise_min_value = self.current_round_raise_min_value
        new_env.current_round_action_dict = copy.deepcopy(self.current_round_action_dict)
        new_env.current_round_value_dict = copy.deepcopy(self.current_round_value_dict)
        new_env.historical_round_action_list = copy.deepcopy(self.historical_round_action_list)
        new_env.historical_round_value_list = copy.deepcopy(self.historical_round_value_list)

        new_env.all_round_player_action_value_dict = copy.deepcopy(self.all_round_player_action_value_dict)

        new_env.info_sets = copy.deepcopy(self.info_sets)
        # new_env在step()前用不上game_infoset，可以置空
        # new_env.game_infoset = copy.deepcopy(self.game_infoset)
        new_env.game_infoset = None

        new_env.pot_value = self.pot_value

        new_env.ignore_all_async_tasks = True

        # 重新随机初始化未知的牌堆，不要重新初始化玩家手牌
        all_known_cards_set = set()

        for player_info_set in new_env.info_sets[self.acting_player_name]:
            for card in player_info_set.player_hand_cards:
                all_known_cards_set.add(card)
        if new_env.current_round >= 1:
            for card in new_env.flop_cards:
                all_known_cards_set.add(card)
        if new_env.current_round >= 2:
            for card in new_env.turn_cards:
                all_known_cards_set.add(card)
        if new_env.current_round >= 3:
            for card in new_env.river_cards:
                all_known_cards_set.add(card)

        all_unknown_cards = list()
        for card in deck:
            if card not in all_known_cards_set:
                all_unknown_cards.append(card)

        num_new_cards = 5 + new_env.num_players * 2 - len(all_known_cards_set)
        new_random_cards = random.sample(all_unknown_cards, num_new_cards)
        for player_name, player_info_set in new_env.info_sets.items():
            if player_name != self.acting_player_name:
                new_player_hand_cards = list()
                for _ in range(2):
                    new_player_hand_cards.append(new_random_cards.pop())
                new_player_hand_cards.sort()
                player_info_set.player_hand_cards = new_player_hand_cards
        if new_env.current_round < 1:
            new_flop_cards = list()
            for _ in range(3):
                new_flop_cards.append(new_random_cards.pop())
            new_flop_cards.sort()
            new_env.flop_cards = new_flop_cards
        if new_env.current_round < 2:
            new_turn_cards = list()
            for _ in range(1):
                new_turn_cards.append(new_random_cards.pop())
            new_env.turn_cards = new_turn_cards
        if new_env.current_round < 3:
            new_river_cards = list()
            for _ in range(1):
                new_river_cards.append(new_random_cards.pop())
            new_env.river_cards = new_river_cards

        return new_env

    def reset_players(self, force_restart_game=False):
        def restart_game():
            self.num_players = self.initial_num_players
            self.init_value = self.initial_init_value

            self.players = dict()
            for i in range(self.num_players):
                self.players[GET_PLAYER_NAME(i)] = DummyAgent(self.init_value, self.small_blind)

            self.button_player_name = GET_PLAYER_NAME(0)

        def restart_when_busted(num_player_value_equals_0):
            self.num_players -= num_player_value_equals_0
            self.init_value = self.initial_init_value * self.initial_num_players / self.num_players

            new_players = dict()
            j = 0
            busted_player_idx_list = []
            for i in range(self.num_players):
                while self.players[GET_PLAYER_NAME(j)].status == PlayerStatus.BUSTED:
                    j += 1
                    busted_player_idx_list.append(j)
                current_player = self.players[GET_PLAYER_NAME(j)]
                current_player.status = PlayerStatus.ONBOARD
                current_player.action = None
                current_player.value_game_start = current_player.value_left
                current_player.value_bet = 0
                current_player.value_win = 0
                new_players[GET_PLAYER_NAME(i)] = current_player
                j += 1
            self.players = new_players

            button_player_idx = GET_PLAYER_ID_BY_NAME(self.button_player_name)
            num_player_busted_before_button = 0
            for busted_player_idx in busted_player_idx_list:
                if busted_player_idx <= button_player_idx:
                    num_player_busted_before_button += 1
            button_player_idx -= num_player_busted_before_button
            self.button_player_name = self.get_next_player_name(GET_PLAYER_NAME(button_player_idx))

        def reset_single_game():
            for player in self.players.values():
                player.status = PlayerStatus.ONBOARD
                player.action = None
                player.value_game_start = player.value_left
                player.value_bet = 0
                player.value_win = 0

            self.button_player_name = self.get_next_player_name(self.button_player_name)

        if force_restart_game or self.players is None:
            # no players inited
            restart_game()
        else:
            num_player_value_equals_0 = 0
            num_player_value_greater_than_0 = 0
            for player in self.players.values():
                if player.value_left <= 0:
                    num_player_value_equals_0 += 1
                else:
                    num_player_value_greater_than_0 += 1

            if num_player_value_greater_than_0 < 2:
                # Already have a winner, game finished
                restart_game()
            elif num_player_value_equals_0 >= 1:
                # One or more players are busted, delete that player and reset game basic info
                restart_when_busted(num_player_value_equals_0)
            else:
                # Just reset a single game for no player busted
                reset_single_game()

    def reset(self, seed=None, cards_dict=None, force_restart_game=False):
        # reset the whole env or just reset a single game
        self.reset_players(force_restart_game=force_restart_game)

        # get player init value for each player
        self.player_init_value_dict = {player_name: player.value_left for player_name, player in self.players.items()}

        # PockerTDA.com规则[34]
        if self.num_players == 2:
            # 单挑局，小盲位翻牌前先行动，翻牌后的每个下注轮后行动
            small_bind_player_name = self.button_player_name
            big_bind_player_name = self.get_next_player_name(small_bind_player_name)
            self.acting_player_name = self.button_player_name
        else:
            # 翻牌前大盲位后先行动，后续按钮位后先行动
            small_bind_player_name = self.get_next_player_name(self.button_player_name)
            big_bind_player_name = self.get_next_player_name(small_bind_player_name)
            self.acting_player_name = self.get_next_player_name(big_bind_player_name)

        small_blind_actual_bet = self.players[small_bind_player_name].set_blinds(self.small_blind)
        big_blind_actual_bet = self.players[big_bind_player_name].set_blinds(self.big_blind)
        blind_round_action_dict = {
            small_bind_player_name: PlayerActions.SMALL_BLIND_RAISE,
            big_bind_player_name: PlayerActions.BIG_BLIND_RAISE,
        }
        blind_round_value_dict = {
            small_bind_player_name: small_blind_actual_bet,
            big_bind_player_name: big_blind_actual_bet,
        }

        self.flop_cards = None
        self.turn_cards = None
        self.river_cards = None

        self.game_over = False
        self.winner = None

        self.current_round = 0
        self.current_round_min_value = max(small_blind_actual_bet, big_blind_actual_bet)
        self.current_round_raise_min_value = max(small_blind_actual_bet, big_blind_actual_bet)
        self.current_round_action_dict = {player_name: blind_round_action_dict.get(player_name, None) for player_name in self.players.keys()}
        self.current_round_value_dict = {player_name: blind_round_value_dict.get(player_name, 0) for player_name in self.players.keys()}
        self.historical_round_action_list = []
        self.historical_round_value_list = []

        # player_name, action, action_value, action_delta_value
        self.all_round_player_action_value_dict = dict()
        self.all_round_player_action_value_dict[self.current_round] = [
            (small_bind_player_name, PlayerActions.SMALL_BLIND_RAISE, small_blind_actual_bet, small_blind_actual_bet),
            (big_bind_player_name, PlayerActions.BIG_BLIND_RAISE, big_blind_actual_bet, max(big_blind_actual_bet - small_blind_actual_bet, small_blind_actual_bet)),
        ]

        self.info_sets = {player_name: InfoSet(player_name, self.player_init_value_dict, self.init_value) for player_name in self.players.keys()}

        self.pot_value = big_blind_actual_bet + small_blind_actual_bet

        self.card_play_init(seed=seed, cards_dict=cards_dict)

        # no more than one ONBOARD player after blind bet
        # 盲注导致allin的情况，不好整合训练数据，暂时认为和模型训练无关，所以直接结束游戏。
        # 这是一个corner case，有bug
        num_player_onboard = 0
        for player in self.players.values():
            if player.status == PlayerStatus.ONBOARD:
                num_player_onboard += 1
        if num_player_onboard <= 1:
            self.finish_game()
            self.reset(force_restart_game=True)

    def card_play_init(self, seed=None, cards_dict=None):
        if seed is not None:
            np.random.seed(seed)

        # Randomly shuffle the deck
        _deck = deck.copy()
        # 避免cards_dict元素指定不完全时发牌重复
        deck_idx_to_drop_list = []
        if cards_dict is not None:
            for cards in cards_dict.values():
                for card in cards:
                    deck_idx_to_drop_list.append(deck_reverse_dict[card])
        deck_idx_to_drop_list.sort(reverse=True)
        for deck_idx in deck_idx_to_drop_list:
            _deck.pop(deck_idx)
        np.random.shuffle(_deck)

        for i, player_name in enumerate(self.players.keys()):
            if cards_dict is not None and player_name in cards_dict and len(cards_dict[player_name]) == 2:
                self.info_sets[player_name].player_hand_cards = sorted(cards_dict[player_name])
            else:
                self.info_sets[player_name].player_hand_cards = sorted(_deck[i * 2: (i + 1) * 2])

        if cards_dict is not None and CARDS_FLOP in cards_dict and len(cards_dict[CARDS_FLOP]) == 3:
            self.flop_cards = sorted(cards_dict[CARDS_FLOP])
        else:
            self.flop_cards = sorted(_deck[self.num_players * 2 + 1: self.num_players * 2 + 4])

        if cards_dict is not None and CARDS_TURN in cards_dict and len(cards_dict[CARDS_TURN]) == 1:
            self.turn_cards = cards_dict[CARDS_TURN].copy()
        else:
            self.turn_cards = sorted(_deck[self.num_players * 2 + 5: self.num_players * 2 + 6])

        if cards_dict is not None and CARDS_RIVER in cards_dict and len(cards_dict[CARDS_RIVER]) == 1:
            self.river_cards = cards_dict[CARDS_RIVER].copy()
        else:
            self.river_cards = sorted(_deck[self.num_players * 2 + 7: self.num_players * 2 + 8])
        self.game_infoset = self.update_get_infoset()

    def update_get_infoset(self):
        acting_player_info_sets = self.info_sets[self.acting_player_name]

        acting_player_info_sets.players = deepcopy(self.players)

        acting_player_info_sets.players_status = self.players[self.acting_player_name].status

        if self.current_round >= 1:
            acting_player_info_sets.flop_cards = self.flop_cards

        if self.current_round >= 2:
            acting_player_info_sets.turn_cards = self.turn_cards

        if self.current_round >= 3:
            acting_player_info_sets.river_cards = self.river_cards

        acting_player_info_sets.current_round = self.current_round

        acting_player_info_sets.all_round_player_action_value_dict = deepcopy(self.all_round_player_action_value_dict)

        current_status_value_left_dict = dict()
        for player_name, player in self.players.items():
            current_status_value_left_dict[player_name] = (player.status, player.value_left, player.value_bet)
        acting_player_info_sets.player_status_value_left_bet_dict = current_status_value_left_dict

        acting_player_info_sets.num_players = self.num_players

        acting_player_info_sets.button_player_name = self.button_player_name

        acting_player_info_sets.pot_value = self.pot_value

        # 每一个桶（下一步行为）的bet value，范围取中值
        acting_player_info_sets.bin_bet_value_list = self.get_bin_value_list_v2()

        return acting_player_info_sets

    def get_next_player_name(self, player_name, ignore_current_player=True):
        if player_name is None:
            return None

        def next_player_name_by_name(name):
            player_id = GET_PLAYER_ID_BY_NAME(name)
            player_id += 1
            if player_id >= self.num_players:
                player_id = 0
            return GET_PLAYER_NAME(player_id)

        next_player_name = next_player_name_by_name(player_name)
        end_player_name = player_name if ignore_current_player else next_player_name
        ignore_first = False if ignore_current_player else True
        while self.players[next_player_name].status != PlayerStatus.ONBOARD:
            next_player_name = next_player_name_by_name(next_player_name)
            # 转了一圈没有PlayerStatus.ONBOARD的人
            if next_player_name == end_player_name:
                if ignore_first:
                    ignore_first = False
                else:
                    next_player_name = None
                    break
        return next_player_name

    def finish_game(self):
        self.game_over = True

        winner_value_dict, player_game_result_dict = self._share_value()
        for player_name, value in winner_value_dict.items():
            player = self.players[player_name]
            player.value_left += value
            player.value_win += value
        for player_name, player_game_result in player_game_result_dict.items():
            player = self.players[player_name]
            player.game_result = player_game_result

        for player in self.players.values():
            player.action = None
            if player.value_left <= 0:
                player.status = PlayerStatus.BUSTED

        return winner_value_dict

    def step(self, action):
        if self.game_over:
            return

        acting_player_name = self.acting_player_name
        current_round_player_historical_value = self.current_round_value_dict[self.acting_player_name]

        current_round = self.current_round
        action = self.players[self.acting_player_name].act(action, self.current_round_min_value, current_round_player_historical_value)

        logging.debug(f'player:{self.acting_player_name} took action:({action[0].name}, {action[1]})')

        if action[0] != PlayerActions.FOLD:
            action_value = action[1]
            if self.players[self.acting_player_name].status == PlayerStatus.ONBOARD:
                assert action_value >= self.current_round_min_value, f'Invalid bet value:{action_value}, must greater or equal than {self.current_round_min_value}'
            self.current_round_raise_min_value = max(self.current_round_raise_min_value, action_value - self.current_round_min_value)
            self.current_round_min_value = max(self.current_round_min_value, action_value)

        self.current_round_action_dict[self.acting_player_name] = action[0]
        self.current_round_value_dict[self.acting_player_name] = action[1]

        # 用于特征拼接
        self.pot_value += action[2]

        # 用于infoset记录游戏进程
        if self.current_round in self.all_round_player_action_value_dict:
            player_action_value_list = self.all_round_player_action_value_dict[self.current_round]
        else:
            player_action_value_list = list()
            self.all_round_player_action_value_dict[self.current_round] = player_action_value_list
        current_player_action_value = (self.acting_player_name, *action)
        player_action_value_list.append(current_player_action_value)

        # 判断一轮结束：(所有玩家都行动过，且(只有一个人raise时没有check 或 没人raise))，或者只有一个人没弃牌
        # 判断一轮结束：(所有玩家都行动过，且(有人raise且所有onboard的玩家raise数都等于当前轮最大raise数 或 没人raise))，或者只有一个人没弃牌
        all_player_finished_once = True
        num_raise_player = 0
        num_onboard_player = 0
        num_onboard_allin_player = 0
        num_fold_player = 0
        all_onboard_check_player_value_valid = True
        for player_name, player in self.players.items():
            if player.status == PlayerStatus.ONBOARD or player.status == PlayerStatus.ALLIN:
                num_onboard_allin_player += 1

                if player.status == PlayerStatus.ONBOARD and player.action is None:
                    all_player_finished_once = False
                elif player.action is not None and player.action[0] == PlayerActions.RAISE:
                    num_raise_player += 1
                if player.status == PlayerStatus.ONBOARD and (player_name not in self.current_round_value_dict or self.current_round_value_dict[player_name] < self.current_round_min_value):
                    all_onboard_check_player_value_valid = False

                if player.status == PlayerStatus.ONBOARD:
                    num_onboard_player += 1
            if player.action is not None and player.action[0] == PlayerActions.FOLD:
                num_fold_player += 1

        # if (all_player_finished_once and ((num_raise_player == 1 and num_check_player == 0) or num_raise_player == 0)) or num_onboard_player <= 1:
        # 本轮正常结束
        is_all_player_finished_valid = (all_player_finished_once and ((num_raise_player >= 1 and all_onboard_check_player_value_valid) or num_raise_player == 0))
        if is_all_player_finished_valid or num_onboard_allin_player <= 1:
            logging.debug(f'round finished')
            self.historical_round_action_list.append(self.current_round_action_dict)
            self.historical_round_value_list.append(self.current_round_value_dict)
            if self.current_round == 3 or num_onboard_allin_player <= 1 or num_onboard_player == 0 or (is_all_player_finished_valid and num_onboard_player <= 1):
                # game finished
                logging.debug(f'game finished')
                if self.settle_automatically:
                    self.finish_game()
                else:
                    self.game_over = True
                logging.debug([f'{player_name}:{int(player.value_left)},{player.status.name}' for player_name, player in self.players.items()])
            else:
                # round finished
                for player in self.players.values():
                    player.reset_action()

                self.acting_player_name = self.get_next_player_name(self.button_player_name, ignore_current_player=False)

                self.current_round_action_dict = {player_name: None for player_name in self.players.keys()}
                self.current_round_value_dict = {player_name: 0 for player_name in self.players.keys()}

                self.current_round += 1
                self.current_round_min_value = 0
                self.current_round_raise_min_value = self.small_blind

                self.game_infoset = self.update_get_infoset()
                logging.debug([f'{player_name}:{int(player.value_left)},{player.status.name}' for player_name, player in self.players.items()])
        else:
            self.acting_player_name = self.get_next_player_name(self.acting_player_name)
            self.game_infoset = self.update_get_infoset()

        return current_round, acting_player_name

    @staticmethod
    def generate_game_result(current_match_winner_set, player_game_result_dict):
        # 以最后一次结算结果为准即可
        if len(current_match_winner_set) > 1:
            current_round_result = GamePlayerResult.EVEN
        else:
            current_round_result = GamePlayerResult.WIN
        for player_name in current_match_winner_set:
            player_game_result_dict[player_name] = current_round_result

    def _share_value(self):
        # player: winning_value
        winner_value_dict = dict()
        # player: game_result
        player_game_result_dict = {player_name: GamePlayerResult.LOSE for player_name in self.players.keys()}
        value_pot = 0
        all_shared_player_name_set = set()
        for action_dict, value_dict in zip(self.historical_round_action_list, self.historical_round_value_list):
            current_round_shared_player_name_set = set()
            current_round_max_value = 0
            for player_name in self.players.keys():
                if player_name in value_dict:
                    player_value = value_dict[player_name]
                    current_round_max_value = max(current_round_max_value, player_value)

            allin_player_value_list = []
            for player_name in self.players.keys():
                if player_name in action_dict and player_name in value_dict:
                    player_action = action_dict[player_name]
                    player_value = value_dict[player_name]
                    if player_action is not None and player_action != PlayerActions.FOLD and player_value < current_round_max_value:
                        allin_player_value_list.append((player_name, player_value))

            spare_value_dict = value_dict.copy()
            if len(allin_player_value_list) > 0:
                # 有allin玩家
                allin_player_value_list.sort(key=itemgetter(1))
                for allin_player_name, _ in allin_player_value_list:
                    # 如果当前player的钱在之前的比牌中已经被分光，不参与后续比牌
                    if allin_player_name not in spare_value_dict or spare_value_dict[allin_player_name] <= 0:
                        current_round_shared_player_name_set.add(allin_player_name)
                        all_shared_player_name_set.add(allin_player_name)
                        continue

                    # 哪些玩家参与本轮比牌
                    current_match_player_cards_dict = dict()
                    # current_match_player_value_dict = dict()
                    for match_player_name in self.players.keys():
                        if match_player_name in action_dict:
                            match_player_action = action_dict[match_player_name]
                            if match_player_action != PlayerActions.FOLD and match_player_name not in current_round_shared_player_name_set:
                                current_match_player_cards_dict[match_player_name] = self.info_sets[match_player_name].player_hand_cards
                                # current_match_player_value_dict[match_player_name] = spare_value_dict[match_player_name]

                    current_match_winner_set = GameEnv.get_winner_set_v2(self.flop_cards, self.turn_cards, self.river_cards, current_match_player_cards_dict)
                    GameEnv.generate_game_result(current_match_winner_set, player_game_result_dict)
                    if allin_player_name in current_match_winner_set:
                        # 本轮比牌allin玩家胜利
                        value_pot, spare_value_dict = self.share_pot_for_winner(allin_player_name, current_match_winner_set, value_pot, spare_value_dict, winner_value_dict)
                    else:
                        value_pot += spare_value_dict[allin_player_name]
                        spare_value_dict.pop(allin_player_name)
                    current_round_shared_player_name_set.add(allin_player_name)
                    all_shared_player_name_set.add(allin_player_name)

            # 把所有没被本轮瓜分的bet加入pot
            for spare_value in spare_value_dict.values():
                value_pot += spare_value

        if value_pot > 0:
            # 所有轮结束的比牌
            action_dict = self.historical_round_action_list[-1]
            current_match_player_cards_dict = dict()
            for match_player_name in self.players.keys():
                match_player_action = action_dict[match_player_name]
                if match_player_action != PlayerActions.FOLD and match_player_name not in all_shared_player_name_set:
                    current_match_player_cards_dict[match_player_name] = self.info_sets[match_player_name].player_hand_cards

            current_match_winner_set = GameEnv.get_winner_set_v2(self.flop_cards, self.turn_cards, self.river_cards, current_match_player_cards_dict)
            GameEnv.generate_game_result(current_match_winner_set, player_game_result_dict)
            value_pot, _ = self.share_pot(current_match_winner_set, value_pot, dict(), winner_value_dict)

        return winner_value_dict, player_game_result_dict

    @staticmethod
    def get_winner_set(flop_cards, turn_cards, river_cards, player_cards_dict):
        if len(player_cards_dict) == 1:
            return {player_name for player_name in player_cards_dict.keys()}

        player_biggest_card_dict = dict()
        board_cards = list(flop_cards)
        GameEnv.merge_sorted_list(board_cards, 3, turn_cards, 1)
        GameEnv.merge_sorted_list(board_cards, 4, river_cards, 1)
        for player_name, hand_cards in player_cards_dict.items():
            biggest_card_dict = GameEnv.get_biggest_cards(board_cards, hand_cards)
            player_biggest_card_dict[player_name] = biggest_card_dict

        winner_player_name_list = list()
        for player_name, hand_cards in player_biggest_card_dict.items():
            if len(winner_player_name_list) == 0:
                winner_player_name_list.append(player_name)
            else:
                historical_winner = winner_player_name_list[0]
                current_winner = GameEnv.compare_cards(historical_winner, player_biggest_card_dict[historical_winner], player_name, hand_cards)
                if current_winner is None:
                    winner_player_name_list.append(player_name)
                elif current_winner != historical_winner:
                    winner_player_name_list.clear()
                    winner_player_name_list.append(current_winner)
        return set(winner_player_name_list)

    # Switch to PokerHandEvaluator(https://github.com/HenryRLee/PokerHandEvaluator/tree/master/python) for better performance.
    @staticmethod
    def get_winner_set_v2(flop_cards, turn_cards, river_cards, player_cards_dict):
        if len(player_cards_dict) == 1:
            return {player_name for player_name in player_cards_dict.keys()}

        public_card_list = [card_to_evaluator(card) for card in flop_cards]
        public_card_list.append(card_to_evaluator(turn_cards[0]))
        public_card_list.append(card_to_evaluator(river_cards[0]))

        lowest_rank = sys.maxsize
        winner_player_name_set = set()
        for player_name, player_hand_cards in player_cards_dict.items():
            hand_card_list = [card_to_evaluator(card) for card in player_hand_cards]
            player_rank = evaluate_cards(*public_card_list, *hand_card_list)
            if player_rank <= lowest_rank:
                if player_rank < lowest_rank:
                    lowest_rank = player_rank
                    winner_player_name_set.clear()
                winner_player_name_set.add(player_name)
        return winner_player_name_set

    def share_pot(self, winner_set, historical_pot, player_value_dict, winner_value_dict):
        spare_player_value_dict = player_value_dict.copy()
        winner_min_value = 0
        for player_name, value in player_value_dict.items():
            if player_name in winner_set:
                if winner_min_value == 0:
                    winner_min_value = value
                else:
                    winner_min_value = min(winner_min_value, value)

        share_value_sum = historical_pot
        for player_name in player_value_dict.keys():
            spare_player_value_dict[player_name] = spare_player_value_dict[player_name] - winner_min_value
            share_value_sum += winner_min_value

        self.share_value_to_winner(share_value_sum, winner_set, winner_value_dict)
        return 0, spare_player_value_dict

    def share_pot_for_winner(self, winner_player_name, winner_set, historical_pot, player_value_dict, winner_value_dict):
        spare_player_value_dict = player_value_dict.copy()
        winner_value = player_value_dict[winner_player_name]

        share_value_sum = historical_pot
        for player_name in player_value_dict.keys():
            spare_value = spare_player_value_dict[player_name]
            if spare_value > winner_value:
                spare_player_value_dict[player_name] = spare_value - winner_value
                share_value_sum += winner_value
            else:
                spare_player_value_dict.pop(player_name)
                share_value_sum += spare_value

        self.share_value_to_winner(share_value_sum, winner_set, winner_value_dict)
        return 0, spare_player_value_dict

    def share_value_to_winner(self, share_value_sum, winner_set, winner_value_dict):
        if share_value_sum > 0:
            share_value_left = share_value_sum
            winner_share_value = floor((share_value_sum / len(winner_set)) // self.small_blind) * self.small_blind
            for winner_player_name in winner_set:
                if winner_player_name in winner_value_dict:
                    winner_value_dict[winner_player_name] = int(winner_value_dict[winner_player_name] + winner_share_value)
                else:
                    winner_value_dict[winner_player_name] = int(winner_share_value)
                share_value_left -= winner_share_value

            # according to the TDA rule: the odd chip goes to the first seat left of the button
            if share_value_left > 0:
                share_player_name = self.button_player_name
                while share_player_name not in winner_set:
                    share_player_name = self.get_next_player_name(self.button_player_name)
                winner_value_dict[share_player_name] = int(winner_value_dict[share_player_name] + share_value_left)

    @staticmethod
    def get_biggest_card_combinations(board_cards, hand_cards):
        def get_one_pair_triple_quatre(card, all_combination_dict):
            one_list = all_combination_dict[CardCombinations.ONES]
            pair_list = all_combination_dict[CardCombinations.PAIRS]
            triple_list = all_combination_dict[CardCombinations.TRIPLE]
            quatre_list = all_combination_dict[CardCombinations.QUATRE]

            if len(one_list) == 0 or one_list[-1].figure != card.figure:
                one_list.append(card)
            elif len(pair_list) == 0 or pair_list[-1].figure != card.figure:
                pair_list.append(card)
            elif len(triple_list) == 0 or triple_list[-1].figure != card.figure:
                triple_list.append(card)
            elif len(quatre_list) == 0 or quatre_list[-1].figure != card.figure:
                quatre_list.append(card)

        def get_straights(card, all_combination_dict):
            straight_list = all_combination_dict[CardCombinations.STRAIGHTS]

            new_straight_list = list()
            need_to_add_single_card = True
            for straight in straight_list:
                if card.figure.value == straight[-1].figure.value:
                    new_straight = straight.copy()[:-1]
                    new_straight.append(card)
                    new_straight_list.append(new_straight)
                    new_straight_list.append(straight)

                    need_to_add_single_card = False
                elif card.figure.value - straight[-1].figure.value == 1:
                    if len(straight) >= 5:
                        new_straight = straight.copy()[-4:]
                        new_straight.append(card)
                        new_straight_list.append(new_straight)
                        new_straight_list.append(straight)
                    else:
                        straight.append(card)
                        new_straight_list.append(straight)

                    need_to_add_single_card = False
                elif len(straight) >= 5:
                    new_straight_list.append(straight)
                    need_to_add_single_card = False

            if need_to_add_single_card:
                new_straight_list.append([card])
            all_combination_dict[CardCombinations.STRAIGHTS] = new_straight_list

        def get_flushes(card, all_combination_dict):
            flush_list = all_combination_dict[CardCombinations.FLUSH]
            if len(flush_list) == 0:
                for _ in CardDecor:
                    flush_list.append(list())
            flush_list[card.decor.value].append(card)

        # t0 = time.time()
        # sort all cards
        all_cards = list(board_cards)
        GameEnv.merge_sorted_list(all_cards, 5, hand_cards, 2)

        # t1 = time.time()
        all_combination_dict = {
            card_combination: list() for card_combination in CardCombinations
        }

        for card in all_cards:
            get_one_pair_triple_quatre(card, all_combination_dict)
            get_straights(card, all_combination_dict)
            get_flushes(card, all_combination_dict)
        # t2 = time.time()

        # post-process for STRAIGHTS
        biggest_straight_flush = None
        biggest_straight = None
        straight_list = all_combination_dict[CardCombinations.STRAIGHTS]
        for straight in straight_list:
            if len(straight) >= 5:
                current_straight = straight[-5:]
                if biggest_straight is None or current_straight[-1].figure.value > biggest_straight[-1].figure.value:
                    biggest_straight = current_straight

                is_straight_flush = True
                decor = current_straight[0].decor
                for check_card in current_straight[1:]:
                    if check_card.decor != decor:
                        is_straight_flush = False
                        break
                if is_straight_flush and (biggest_straight_flush is None or current_straight[-1].figure.value > biggest_straight_flush[-1].figure.value):
                    biggest_straight_flush = current_straight
        if biggest_straight_flush is not None:
            all_combination_dict[CardCombinations.STRAIGHTS] = (biggest_straight_flush, True)
        elif biggest_straight is not None:
            all_combination_dict[CardCombinations.STRAIGHTS] = (biggest_straight, False)
        else:
            all_combination_dict[CardCombinations.STRAIGHTS] = (list(), False)

        # post-process for FLUSH
        actual_decor_list = []
        for decor_list in all_combination_dict[CardCombinations.FLUSH]:
            if len(decor_list) >= 5:
                actual_decor_list = decor_list[-5:]
                break
        all_combination_dict[CardCombinations.FLUSH] = actual_decor_list

        # t3 = time.time()
        # logging.info(f't0:{t1 - t0}, t1:{t2 - t1}, t2:{t3 - t2}')
        return all_combination_dict

    @staticmethod
    def get_biggest_cards(board_cards, hand_cards):
        all_combination_dict = GameEnv.get_biggest_card_combinations(board_cards, hand_cards)

        biggest_combination_dict = dict()
        straight_list, is_straight_flush = all_combination_dict[CardCombinations.STRAIGHTS]
        flush_list = all_combination_dict[CardCombinations.FLUSH]
        # straight flush
        if is_straight_flush:
            biggest_combination_dict[CardCombinations.STRAIGHTS] = list(straight_list)
            biggest_combination_dict[CardCombinations.FLUSH] = list(straight_list)
            return biggest_combination_dict
        else:
            # flush
            if len(all_combination_dict[CardCombinations.FLUSH]) > 0:
                biggest_combination_dict[CardCombinations.FLUSH] = list(flush_list)
                return biggest_combination_dict

            # straight
            if len(straight_list) > 0:
                biggest_combination_dict[CardCombinations.STRAIGHTS] = list(straight_list)
                return biggest_combination_dict

        cards_left = 5
        quatre_list = all_combination_dict[CardCombinations.QUATRE]
        triple_list = all_combination_dict[CardCombinations.TRIPLE]
        pairs_list = all_combination_dict[CardCombinations.PAIRS]
        ones_list = all_combination_dict[CardCombinations.ONES]
        if len(quatre_list) > 0:
            biggest_combination_dict[CardCombinations.QUATRE] = list(quatre_list)
            cards_left -= 4
        if cards_left >= 3 and len(triple_list) > 0:
            biggest_combination_dict[CardCombinations.TRIPLE] = list(triple_list[-1:])
            cards_left -= 3
        if cards_left >= 2 and len(pairs_list) > 0:
            for card in pairs_list[::-1]:
                if cards_left >= 2 and (CardCombinations.TRIPLE not in biggest_combination_dict or card.figure != biggest_combination_dict[CardCombinations.TRIPLE][0].figure):
                    if CardCombinations.PAIRS not in biggest_combination_dict:
                        biggest_combination_dict[CardCombinations.PAIRS] = [card]
                    else:
                        biggest_combination_dict[CardCombinations.PAIRS].insert(0, card)
                    cards_left -= 2
        if cards_left >= 1 and len(ones_list) > 0:
            for card in ones_list[::-1]:
                if cards_left >= 1:
                    card_need_to_add = True
                    if CardCombinations.PAIRS in biggest_combination_dict:
                        for exist_card in biggest_combination_dict[CardCombinations.PAIRS]:
                            if exist_card.figure == card.figure:
                                card_need_to_add = False
                                break
                    if card_need_to_add:
                        if CardCombinations.ONES not in biggest_combination_dict:
                            biggest_combination_dict[CardCombinations.ONES] = [card]
                        else:
                            biggest_combination_dict[CardCombinations.ONES].insert(0, card)
                        cards_left -= 1
        return biggest_combination_dict

    @staticmethod
    def compare_cards(name1, card_dict1, name2, card_dict2):
        for compare_rule_list in GameEnv.COMPARE_RULES:
            all_rules_in_card_dict1 = True
            all_rules_in_card_dict2 = True
            for compare_rule in compare_rule_list:
                if compare_rule not in card_dict1:
                    all_rules_in_card_dict1 = False
                if compare_rule not in card_dict2:
                    all_rules_in_card_dict2 = False
            if all_rules_in_card_dict1 and not all_rules_in_card_dict2:
                return name1
            elif not all_rules_in_card_dict1 and all_rules_in_card_dict2:
                return name2
            elif all_rules_in_card_dict1 and all_rules_in_card_dict2:
                # 每条规则，逐一比较
                for compare_rule in compare_rule_list:
                    cards_in_rule1 = card_dict1[compare_rule]
                    cards_in_rule2 = card_dict2[compare_rule]
                    cards_in_rule_len1 = len(cards_in_rule1)
                    cards_in_rule_len2 = len(cards_in_rule2)
                    if cards_in_rule_len1 > cards_in_rule_len2:
                        return name1
                    elif cards_in_rule_len1 < cards_in_rule_len2:
                        return name2
                    else:
                        for card1, card2 in zip(cards_in_rule1[::-1], cards_in_rule2[::-1]):
                            if card1.figure.value > card2.figure.value:
                                return name1
                            elif card1.figure.value < card2.figure.value:
                                return name2
        return None

    @staticmethod
    def merge_sorted_list(nums1, m: int, nums2, n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        sorted = []
        p1, p2 = 0, 0
        while p1 < m or p2 < n:
            if p1 == m:
                sorted.append(nums2[p2])
                p2 += 1
            elif p2 == n:
                sorted.append(nums1[p1])
                p1 += 1
            elif nums1[p1] < nums2[p2]:
                sorted.append(nums1[p1])
                p1 += 1
            else:
                sorted.append(nums2[p2])
                p2 += 1
        nums1[:] = sorted

    def get_valid_action_info(self):
        action_mask_list = []
        action_value_or_ranges_list = []
        acting_player_agent = self.players[self.acting_player_name]
        current_round_acting_player_historical_value = self.current_round_value_dict[self.acting_player_name]
        delta_value_to_call = self.current_round_min_value - current_round_acting_player_historical_value
        delta_min_value_to_raise = self.current_round_min_value + self.current_round_raise_min_value - current_round_acting_player_historical_value
        delta_min_value_proportion_to_raise = delta_min_value_to_raise / acting_player_agent.value_left
        for player_action, (range_start, range_end) in ACTION_BINS_DICT:
            if player_action == PlayerActions.RAISE:
                if range_start == 1.:
                    action_mask_list.append(False)
                    if acting_player_agent.value_left <= delta_value_to_call:
                        action_value_or_ranges_list.append((PlayerActions.CHECK_CALL, acting_player_agent.value_left + current_round_acting_player_historical_value))
                    else:
                        action_value_or_ranges_list.append((PlayerActions.RAISE, acting_player_agent.value_left + current_round_acting_player_historical_value))
                elif delta_min_value_proportion_to_raise < range_start:
                    action_mask_list.append(False)
                    action_value_or_ranges_list.append((PlayerActions.RAISE, (range_start, range_end)))
                elif delta_min_value_proportion_to_raise <= range_end:
                    action_mask_list.append(False)
                    action_value_or_ranges_list.append((PlayerActions.RAISE, (delta_min_value_proportion_to_raise, range_end)))
                else:
                    action_mask_list.append(True)
                    action_value_or_ranges_list.append(None)
            elif player_action == PlayerActions.CHECK_CALL:
                if delta_value_to_call < acting_player_agent.value_left:
                    action_mask_list.append(False)
                    action_value_or_ranges_list.append((PlayerActions.CHECK_CALL, self.current_round_min_value))
                else:
                    action_mask_list.append(True)
                    action_value_or_ranges_list.append(None)
            else:
                action_mask_list.append(False)
                action_value_or_ranges_list.append((PlayerActions.FOLD, 0))
        return action_mask_list, action_value_or_ranges_list, acting_player_agent.value_left, current_round_acting_player_historical_value

    def get_valid_action_info_v2(self):
        action_mask_list = []
        action_value_or_ranges_list = []
        acting_player_agent = self.players[self.acting_player_name]
        current_round_acting_player_historical_value = self.current_round_value_dict[self.acting_player_name]
        delta_value_to_call = self.current_round_min_value - current_round_acting_player_historical_value
        delta_min_value_to_raise = self.current_round_min_value + self.current_round_raise_min_value - current_round_acting_player_historical_value
        is_raise_range_valid = delta_min_value_to_raise < acting_player_agent.value_left
        for player_action, (range_start, range_end) in ACTION_BINS_DICT:
            if player_action == PlayerActions.RAISE:
                if range_start == 1.:
                    action_mask_list.append(False)
                    if acting_player_agent.value_left <= delta_value_to_call:
                        action_value_or_ranges_list.append((PlayerActions.CHECK_CALL, acting_player_agent.value_left + current_round_acting_player_historical_value))
                    else:
                        action_value_or_ranges_list.append((PlayerActions.RAISE, acting_player_agent.value_left + current_round_acting_player_historical_value))
                elif is_raise_range_valid:
                    action_mask_list.append(False)
                    action_value_or_ranges_list.append((PlayerActions.RAISE, (range_start, range_end)))
                else:
                    action_mask_list.append(True)
                    action_value_or_ranges_list.append(None)
            elif player_action == PlayerActions.CHECK_CALL:
                if delta_value_to_call < acting_player_agent.value_left:
                    action_mask_list.append(False)
                    action_value_or_ranges_list.append((PlayerActions.CHECK_CALL, self.current_round_min_value))
                else:
                    action_mask_list.append(True)
                    action_value_or_ranges_list.append(None)
            else:
                action_mask_list.append(False)
                action_value_or_ranges_list.append((PlayerActions.FOLD, 0))
        return action_mask_list, action_value_or_ranges_list, acting_player_agent.value_left, current_round_acting_player_historical_value, delta_min_value_to_raise

    def get_bin_value_list_v2(self):
        bin_bet_value_list = list()
        acting_player_value_left = self.players[self.acting_player_name].value_left
        call_delta_min_value = self.current_round_min_value - self.current_round_value_dict[self.acting_player_name]
        raise_delta_min_value = self.current_round_min_value + self.current_round_raise_min_value - self.current_round_value_dict[self.acting_player_name]
        is_call_valid = call_delta_min_value > 0
        is_raise_valid = raise_delta_min_value > 0
        raise_delta_range_max_value = acting_player_value_left - raise_delta_min_value
        # fold和allin不计算分桶特征，因为都包含在了其他特征内
        for player_action, (range_start, range_end) in ACTION_BINS_DICT:
            if player_action == PlayerActions.CHECK_CALL:
                if is_call_valid:
                    bin_bet_value_list.append(call_delta_min_value)
                else:
                    bin_bet_value_list.append(-1)
            elif player_action == PlayerActions.RAISE:
                if range_start < 1.:
                    if is_raise_valid:
                        bin_bet_value_list.append(GET_VALID_BET_VALUE(raise_delta_range_max_value * (range_end + range_start) / 2, self.small_blind) + raise_delta_min_value)
                    else:
                        bin_bet_value_list.append(-1)
        return bin_bet_value_list

class InfoSet(object):
    """
    The game state is described as infoset, which
    includes all the information in the current situation,
    such as the hand cards of the three players, the
    historical moves, etc.
    """
    def __init__(self, player_name, player_init_value_dict, game_init_value):
        # The player name, i.e., player_0, player_1, etc.
        self.player_name = player_name
        # Player init value for current game.
        self.player_value_init_dict = player_init_value_dict
        # Game init value.
        self.game_init_value = game_init_value
        # The hand cards of the current player. A list.
        self.player_hand_cards = None
        # The status of each player, i.e., ONBOARD, FOLDED, BUSTED. It is a dict with str-->int
        self.players_status = None
        # The flop round cards. A list.
        self.flop_cards = None
        # The turn round card. A list.
        self.turn_cards = None
        # The river round card. A list.
        self.river_cards = None
        # Current round
        self.current_round = None
        # All moves of all players in historical rounds. It is a dict with str-->(int, int)
        self.all_round_player_action_value_dict = None
        # Current status of all players. A dict
        self.player_status_value_left_bet_dict = None
        # Number of players. A str
        self.num_players = None
        # The button player name. A str
        self.button_player_name = None
        # The total pot value. A float
        self.pot_value = None
        # All bins (Call, Bets) bet value. A list
        self.bin_bet_value_list = None
