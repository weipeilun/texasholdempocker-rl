import logging
import time
import traceback
import numpy as np
from tools.biner import *
from .cards import *
from .game import GameEnv, deck

class Env:
    """
    Doudizhu multi-agent wrapper
    """
    def __init__(self, winning_probability_generating_task_queue, num_bins, num_players: int = MAX_PLAYER_NUMBER, init_value: int = 100_000, small_blind=25, big_blind=50, num_player_fields=10, game_env=None, ignore_all_async_tasks=False):
        """
        Here, we use dummy agents.
        This is because, in the orignial game, the players
        are `in` the game. Here, we want to isolate
        players and environments to have a more gym style
        interface. To achieve this, we use dummy players
        to play. For each move, we tell the corresponding
        dummy player which action to play, then the player
        will perform the actual action in the game engine.
        """
        self.winning_probability_generating_task_queue = winning_probability_generating_task_queue
        self.num_bins = num_bins
        self.ignore_all_async_tasks = ignore_all_async_tasks

        self.num_players = num_players
        assert num_players <= MAX_PLAYER_NUMBER, f'There are up to {MAX_PLAYER_NUMBER} players in the game, received {num_players}'

        self.init_value = init_value
        self.small_blind = small_blind
        self.big_blind = big_blind

        self.num_player_fields = num_player_fields

        # Initialize the internal environment
        if game_env is None:
            self._env = GameEnv(self.num_players, self.init_value, small_blind=self.small_blind, big_blind=self.big_blind)
        else:
            self._env = game_env

        # 使用round_num + player_name作为key过滤任务，减少无用计算
        self.reward_cal_task_set = set()

        # 初始化连续特征的分桶器
        bin_cutter_interval = 1 / (num_bins - 1)
        self.player_init_value_to_pool_cutter = CutByThreshold(np.arange(0, MAX_PLAYER_NUMBER, MAX_PLAYER_NUMBER * bin_cutter_interval))
        self.player_value_left_to_pool_cutter = CutByThreshold(np.arange(0, MAX_PLAYER_NUMBER, MAX_PLAYER_NUMBER * bin_cutter_interval))
        self.player_value_left_to_player_cutter = CutByThreshold(np.arange(0, 1, bin_cutter_interval))

        self.call_min_value_to_game_init_value_cutter = CutByThreshold(np.arange(0, MAX_PLAYER_NUMBER, MAX_PLAYER_NUMBER * bin_cutter_interval))
        # 以下，因为call_min_value可能大于分母，而超过分母的值都意味着player要allin，风险极大，则着重刻画小于1和大于1的部分
        self.call_min_value_to_player_init_value_cutter = CutByThreshold(np.arange(0, 1 + bin_cutter_interval, bin_cutter_interval))
        self.call_min_value_to_player_value_left_cutter = CutByThreshold(np.arange(0, 1 + bin_cutter_interval, bin_cutter_interval))

        self.raise_min_value_to_game_init_value_cutter = CutByThreshold(np.arange(0, MAX_PLAYER_NUMBER, MAX_PLAYER_NUMBER * bin_cutter_interval))
        # 以下，因为raise_min_value可能大于分母，而超过分母的值都意味着player要allin，风险极大，则着重刻画小于1和大于1的部分
        self.raise_min_value_to_player_init_value_cutter = CutByThreshold(np.arange(0, 1 + bin_cutter_interval, bin_cutter_interval))
        self.raise_min_value_to_player_value_left_cutter = CutByThreshold(np.arange(0, 1 + bin_cutter_interval, bin_cutter_interval))

        self.game_id = None

    def reset(self, game_id, seed=None):
        """
        Every time reset is called, the environment
        will be re-initialized with a new deck of cards.
        This function is usually called when a game is over.
        """
        logging.debug(f'env reset')
        self.game_id = game_id

        # reset environment
        self._env.reset(seed=seed)

        logging.debug([f'{player_name}:{int(player.value_left)},{player.status.name}' for player_name, player in self._env.players.items()])

        infoset = self._game_infoset

        return self.get_obs(infoset, is_last_round=False), {}

    # 修改经典step框架，异步计算reward
    def step(self, action):
        """
        Step function takes as input the action, which
        is a list of integers, and output the next obervation,
        reward, and a Boolean variable indicating whether the
        current game is finished. It also returns an empty
        dictionary that is reserved to pass useful information.
        """
        acted_round_num, acted_player_name = self._env.step(action)

        info = dict()
        info[KEY_ROUND_NUM] = acted_round_num
        info[KEY_ACTED_PLAYER_NAME] = acted_player_name

        task_key = (acted_round_num, acted_player_name)
        if not self.ignore_all_async_tasks and task_key not in self.reward_cal_task_set:
            player_hand_card = self._env.info_sets[acted_player_name].player_hand_cards
            game_infoset = self._env.game_infoset
            self._gen_cal_reward_task(acted_player_name, acted_round_num, player_hand_card, game_infoset)

            self.reward_cal_task_set.add(task_key)

        if self._game_over:
            done = True
            reward = self._get_final_reward()
            logging.debug(f'reward={reward}')
            # obs = self.get_obs(self._env.info_sets, is_last_round=True)
            obs = None

            if not self.ignore_all_async_tasks:
                self.reward_cal_task_set.clear()
        else:
            done = False
            # reward = self._get_reward(game_id, step_id, acted_player_name, self.reward_cal_task_dict)
            reward = None
            obs = self.get_obs(self._game_infoset, is_last_round=False)
        return obs, reward, done, info

    def new_random(self):
        return Env(winning_probability_generating_task_queue=None,
                   num_bins=self.num_bins,
                   num_players=self.num_players,
                   init_value=self.init_value,
                   small_blind=self.small_blind,
                   big_blind=self.big_blind,
                   game_env=self._env.new_random(),
                   ignore_all_async_tasks=True)

    def close(self):
        pass

    # 注意这个方法计算的reward是考虑胜率后的，相对于玩家当前value的reward
    def _get_reward(self, game_id, step_id, player_name, reward_cal_task_dict):
        """
        This function is called in the end of each
        game. It returns either 1/-1 for win/loss,
        or ADP, i.e., every bomb will double the score.
        """
        player_status = self._env.players[player_name]
        if player_status.status == PlayerStatus.ONBOARD or player_status.status == PlayerStatus.ALLIN:
            num_other_players = 0
            for other_player_name, other_player_status in self._env.players.items():
                if player_name != other_player_name and (other_player_status.status == PlayerStatus.ONBOARD or other_player_status.status == PlayerStatus.ALLIN):
                    num_other_players += 1
            if num_other_players == 0:
                logging.warning(f'No other players found when get reward. Should not come into this block.')
                player_value_bet = player_status.value_bet
                player_value_game_start = player_status.value_game_start
                return player_value_bet / player_value_game_start
            return None
        else:
            player_value_bet = player_status.value_bet
            player_value_game_start = player_status.value_game_start
            return - player_value_bet / player_value_game_start

    def _get_final_reward(self):
        """
        This function is called in the end of each
        game. It returns either 1/-1 for win/loss,
        or ADP, i.e., every bomb will double the score.
        """
        all_player_final_reward = dict()
        for player_name, player in self._env.players.items():
            game_result_value = player.game_result.value
            net_win_value = player.value_win - player.value_bet
            reward_value = net_win_value / player.value_game_start
            all_player_final_reward[player_name] = (game_result_value, reward_value, net_win_value)
        return all_player_final_reward

    def _gen_cal_reward_task(self, player_name, current_round, player_hand_card, game_infoset):
        assert self.game_id is not None, 'game_id should not be None.'

        all_known_cards = set()
        for hand_card in player_hand_card:
            all_known_cards.add(hand_card)

        flop_cards = None
        turn_cards = None
        river_cards = None
        if current_round >= 1:
            flop_cards = game_infoset.flop_cards
            for flop_card in game_infoset.flop_cards:
                all_known_cards.add(flop_card)

        if current_round >= 2:
            turn_cards = game_infoset.turn_cards
            for turn_card in game_infoset.turn_cards:
                all_known_cards.add(turn_card)

        if current_round >= 3:
            river_cards = game_infoset.river_cards
            for river_card in game_infoset.river_cards:
                all_known_cards.add(river_card)

        all_unknown_cards = []
        for card in deck:
            if card not in all_known_cards:
                all_unknown_cards.append(card)

        # self.generate_reward_cal_task_recurrent(self.game_id, player_name, current_round, flop_cards, turn_cards, river_cards, player_hand_card)
        self.winning_probability_generating_task_queue.put((self.game_id, player_name, current_round, flop_cards, turn_cards, river_cards, player_hand_card))

    # 注意这个方法暂只支持生成两个player的对局，因为所算的“客观胜率”就是指两个player的对局
    def generate_reward_cal_task_recurrent(self, game_id, player_name, current_round,
                                           flop_cards, turn_cards, river_cards, player_hand_card):
        def cal_unordered_combination(num_all, num_choice):
            result = 1.
            for i in range(num_choice):
                result *= (num_all - i)
                result /= (i + 1)
            return int(result)

        def generate_recurrent(generate_recurrent_and_randomly_num_list, exist_card_list, current_round_deck_idx):
            num_generated = 0
            current_round_gen_num, num_random_generates, is_segment_end = generate_recurrent_and_randomly_num_list.pop(0)
            num_cards_to_generate_all_other_round = 0
            if not is_segment_end:
                for round_gen_num, _, segment_end in generate_recurrent_and_randomly_num_list:
                    num_cards_to_generate_all_other_round += round_gen_num
                    if segment_end:
                        break

            is_last_round = len(generate_recurrent_and_randomly_num_list) == 0
            exist_card_set = set(exist_card_list)
            if current_round_gen_num == 1:
                for deck_idx in range(current_round_deck_idx, NUM_CARDS):
                    current_card = deck[deck_idx]
                    if current_card not in exist_card_set:
                        new_exist_card_list = exist_card_list.copy()
                        new_exist_card_list.append(current_card)
                        if is_last_round:
                            self.reward_batch_data_queue.put(new_exist_card_list)
                            num_generated += 1
                        elif num_cards_to_generate_all_other_round + deck_idx < NUM_CARDS:
                            new_round_gen_num_list = generate_recurrent_and_randomly_num_list.copy()
                            if is_segment_end:
                                new_current_round_deck_idx = 0
                            else:
                                new_current_round_deck_idx = deck_idx + 1
                            num_generated += generate_recurrent(new_round_gen_num_list, new_exist_card_list, new_current_round_deck_idx)
            else:
                all_cards_not_exist_list = list()
                for deck_idx in range(0, NUM_CARDS):
                    current_card = deck[deck_idx]
                    if current_card not in exist_card_set:
                        all_cards_not_exist_list.append(current_card)
                all_cards_not_exist_array = np.array(all_cards_not_exist_list)
                # 可随机生成的无序组合数大于本回合应生成的牌局数才有意义
                num_cards_not_exist_combination = cal_unordered_combination(len(all_cards_not_exist_array), current_round_gen_num)
                if num_cards_not_exist_combination >= num_random_generates:
                    for _ in range(num_random_generates):
                        generated_cards = exist_card_list.copy()
                        cards_choice = np.random.choice(all_cards_not_exist_array, current_round_gen_num)
                        generated_cards.extend(cards_choice)
                        self.reward_batch_data_queue.put(generated_cards)
                        num_generated += 1
                else:
                    raise ValueError(f'Card combination generator error: can not generate {num_random_generates} cards randomly if card choice is {len(all_cards_not_exist_array)}')

            return num_generated

        # current_round: [(current_round_gen_num, num_random_generates, is_segment_end)]
        # 第一轮只遍历flop轮，其他随机，控制在10w次模拟(19600 * 5)
        # 第二轮只遍历turn轮，其他随机，控制在10w次模拟(2162 * 45)
        # 第三轮全部遍历，模拟次数(990 * 46)
        # 第四轮全部遍历，模拟次数(990)
        generate_recurrent_and_randomly_num_list_dict = {
            0: ([(1, 0, False), (1, 0, False), (1, 0, True), (4, 5, True)], 98000),
            1: ([(1, 0, True), (1, 0, True), (2, 45, True)], 97290),
            2: ([(1, 0, True), (1, 0, False), (1, 0, True)], 45540),
            3: ([(1, 0, False), (1, 0, True)], 990)
        }
        generate_recurrent_and_randomly_num_list, num_estimation = generate_recurrent_and_randomly_num_list_dict[current_round]
        self.reward_batch_signal_queue.put((game_id, player_name, current_round, num_estimation))

        exist_card_list = player_hand_card.copy()
        if current_round > 0:
            for card in flop_cards:
                exist_card_list.append(card)
        if current_round > 1:
            for card in turn_cards:
                exist_card_list.append(card)
        if current_round > 2:
            for card in river_cards:
                exist_card_list.append(card)
        num_data_generated = generate_recurrent(generate_recurrent_and_randomly_num_list, exist_card_list, 0)
        assert num_data_generated == num_estimation, f'Actual number of data generated:{num_data_generated} not equals to num_estimation:{num_estimation}, multiprocess calculation will miscalculate.'

    def _get_reward_value(self):
        """
        This function is called in the end of each
        game. It returns either 1/-1 for win/loss,
        or ADP, i.e., every bomb will double the score.
        """
        player_role_reward_value = dict()
        for player_name, init_value in self._env.player_init_value_dict.items():
            player = self._env.players[player_name]

            player_role = get_player_role(player_name, self._env.button_player_name, self._env.num_players)
            player_role_reward_value[player_role] = (player.value_left - init_value) / init_value
        return player_role_reward_value

    @property
    def _game_infoset(self):
        """
        Here, inforset is defined as all the information
        in the current situation, incuding the hand cards
        of all the players, all the historical moves, etc.
        That is, it contains perferfect infomation. Later,
        we will use functions to extract the observable
        information from the views of the three players.
        """
        return self._env.game_infoset

    @property
    def _game_bomb_num(self):
        """
        The number of bombs played so far. This is used as
        a feature of the neural network and is also used to
        calculate ADP.
        """
        return self._env.get_bomb_num()

    @property
    def _game_winner(self):
        """ A string of landlord/peasants
        """
        return self._env.get_winner()

    @property
    def _acting_player_name(self):
        """
        The player that is active. It can be landlord,
        landlod_down, or landlord_up.
        """
        return self._env.acting_player_name

    @property
    def _game_over(self):
        """ Returns a Boolean
        """
        return self._env.game_over

    def _map_obs_idx_to_model_index(self, current_round, current_player_role, cards, sorted_cards, current_all_player_status):
        sorted_item_list = list()
        sorted_item_list.append(current_round)
        sorted_item_list.append(current_player_role)

        all_figure, all_decor = list(), list()
        for card_representation_list in cards:
            for card_representation in card_representation_list:
                all_figure.append(card_representation[0])
                all_decor.append(card_representation[1])
        for card_representation in sorted_cards:
            all_figure.append(card_representation[0])
            all_decor.append(card_representation[1])
        sorted_item_list.extend(all_figure)
        sorted_item_list.extend(all_decor)

        for i in range(self.num_player_fields):
            for j in range(MAX_PLAYER_NUMBER):
                sorted_item_list.append(current_all_player_status[j][i])

        return np.array(sorted_item_list, dtype=np.int32)

    def get_obs(self, infoset, is_last_round):
        """
        This function obtains observations with imperfect information
        from the infoset. It has three branches since we encode
        different features for different positions.

        This function will return dictionary named `obs`. It contains
        several fields. These fields will be used to train the model.
        One can play with those features to improve the performance.

        `position` is a string that can be landlord/landlord_down/landlord_up

        `x_batch` is a batch of features (excluding the hisorical moves).
        It also encodes the action feature

        `z_batch` is a batch of features with hisorical moves only.

        `legal_actions` is the legal moves

        `x_no_action`: the features (exluding the hitorical moves and
        the action features). It does not have the batch dim.

        `z`: same as z_batch but not a batch.
        """
        # todo: 此处先把历史下注行为干掉，没想好要怎么对玩家位置编码
        if is_last_round:
            # settle round
            current_round = 5
            # # `not a player`
            # current_player_role = MAX_PLAYER_NUMBER

            obs_dict = dict()
            for player_name, _infoset in infoset.items():
                if _infoset.player_name is not None and _infoset.button_player_name is not None:
                    current_player_role = get_player_role(_infoset.player_name, _infoset.button_player_name, _infoset.num_players)

                    hand_cards = get_card_representations(_infoset.player_hand_cards, 2)
                    flop_cards = get_card_representations(_infoset.flop_cards, 3)
                    turn_cards = get_card_representations(_infoset.turn_cards, 1)
                    river_cards = get_card_representations(_infoset.river_cards, 1)

                    sorted_cards = sort_card_representations(hand_cards, flop_cards, turn_cards, river_cards)

                    current_all_player_status = self.get_all_player_current_status(_infoset)

                    obs_dict[player_name] = (current_round, current_player_role,
                    [hand_cards, flop_cards, turn_cards, river_cards], sorted_cards, current_all_player_status)
            return obs_dict
        else:
            current_round = infoset.current_round
            current_player_role = get_player_role(infoset.player_name, infoset.button_player_name, infoset.num_players)

            hand_cards = get_card_representations(infoset.player_hand_cards, 2)
            flop_cards = get_card_representations(infoset.flop_cards, 3)
            turn_cards = get_card_representations(infoset.turn_cards, 1)
            river_cards = get_card_representations(infoset.river_cards, 1)

            sorted_cards = sort_card_representations(hand_cards, flop_cards, turn_cards, river_cards)

            current_all_player_status = self.get_all_player_current_status(infoset)

        #     all_historical_player_action = get_all_historical_action(infoset)
        #
        # return (current_round, current_player_role,
        #     [hand_cards, flop_cards, turn_cards, river_cards],
        #     current_all_player_status, all_historical_player_action)

            # return (current_round, current_player_role,
            #     [hand_cards, flop_cards, turn_cards, river_cards],
            #     sorted_cards, current_all_player_status)
        return self._map_obs_idx_to_model_index(current_round, current_player_role,
                [hand_cards, flop_cards, turn_cards, river_cards],
                sorted_cards, current_all_player_status)

    def get_all_player_current_status(self, infoset):
        if infoset is None:
            all_player_current_status_list = []
        else:
            all_player_current_status_list = [None] * infoset.num_players

            player_name = infoset.player_name
            game_init_value = infoset.game_init_value
            call_min_value = infoset.call_min_value
            raise_min_value = infoset.raise_min_value
            for current_player_name, player_init_value in infoset.player_value_init_dict.items():
                player_status, player_value_left = infoset.player_status_value_left_dict[current_player_name]
                # 这用对于当前玩家的相对位置
                player_role = get_player_role(current_player_name, player_name, infoset.num_players)

                # 提示局势整体的激进/保守偏好
                player_init_value_to_pool = player_init_value / game_init_value
                # 提示当前行为的绝对参照性的激进/保守偏好
                player_value_left_to_pool = player_value_left / game_init_value
                # 提示当前行为的局势相对性的激进/保守偏好
                player_value_left_to_player = player_value_left / player_init_value
                # 提示当前最小跟注行为的绝对参照风险
                call_min_value_to_game_init_value = call_min_value / game_init_value
                # 提示当前最小跟注行为的相对局势风险
                call_min_value_to_player_init_value = call_min_value / player_init_value
                # 提示当前最小跟注行为对玩家离场的风险
                call_min_value_to_player_value_left = call_min_value / player_value_left if player_value_left > 0 else MAX_PLAYER_NUMBER
                # 提示当前最小加注行为的绝对参照风险
                raise_min_value_to_game_init_value = raise_min_value / game_init_value
                # 提示当前最小加注行为的相对局势风险
                raise_min_value_to_player_init_value = raise_min_value / player_init_value
                # 提示当前最小加注行为对玩家离场的风险
                raise_min_value_to_player_value_left = raise_min_value / player_value_left if player_value_left > 0 else MAX_PLAYER_NUMBER

                # 分桶
                player_init_value_to_pool_bin = self.player_init_value_to_pool_cutter.cut(player_init_value_to_pool)
                player_value_left_to_pool_bin = self.player_value_left_to_pool_cutter.cut(player_value_left_to_pool)
                player_value_left_to_player_bin = self.player_value_left_to_player_cutter.cut(player_value_left_to_player)
                call_min_value_to_game_init_value_bin = self.call_min_value_to_game_init_value_cutter.cut(call_min_value_to_game_init_value)
                call_min_value_to_player_init_value_bin = self.call_min_value_to_player_init_value_cutter.cut(call_min_value_to_player_init_value)
                call_min_value_to_player_value_left_bin = self.call_min_value_to_player_value_left_cutter.cut(call_min_value_to_player_value_left)
                raise_min_value_to_game_init_value_bin = self.raise_min_value_to_game_init_value_cutter.cut(raise_min_value_to_game_init_value)
                raise_min_value_to_player_init_value_bin = self.raise_min_value_to_player_init_value_cutter.cut(raise_min_value_to_player_init_value)
                raise_min_value_to_player_value_left_bin = self.raise_min_value_to_player_value_left_cutter.cut(raise_min_value_to_player_value_left)

                all_player_current_status_list[player_role] = (player_status.value,
                                                               player_init_value_to_pool_bin,
                                                               player_value_left_to_pool_bin,
                                                               player_value_left_to_player_bin,
                                                               call_min_value_to_game_init_value_bin,
                                                               call_min_value_to_player_init_value_bin,
                                                               call_min_value_to_player_value_left_bin,
                                                               raise_min_value_to_game_init_value_bin,
                                                               raise_min_value_to_player_init_value_bin,
                                                               raise_min_value_to_player_value_left_bin)
        return all_player_current_status_list


def get_card_representations(list_cards, num_cards):
    """
    A utility function that transforms the actions, i.e.,
    A list of integers into card matrix. Here we remove
    the six entries that are always zero and flatten the
    the representations.
    """
    card_representation_list = [[len(CardFigure), len(CardDecor)] for _ in range(num_cards)]
    if list_cards is not None:
        list_cards_len = len(list_cards)
        assert list_cards_len == num_cards, f'Length of list_cards does not equal to num_cards.'

        for card, card_representation in zip(list_cards, card_representation_list):
            card_representation[0] = card.figure.value
            card_representation[1] = card.decor.value
    return card_representation_list


def sort_card_representations(hand_cards, flop_cards, turn_cards, river_cards):
    """
    A utility function that transforms the actions, i.e.,
    A list of integers into card matrix. Here we remove
    the six entries that are always zero and flatten the
    the representations.
    """
    card_list = list()
    card_list.extend(hand_cards)
    card_list.extend(flop_cards)
    card_list.extend(turn_cards)
    card_list.extend(river_cards)
    card_list.sort()
    return card_list


def get_player_role(player_name, king_player_name, num_players):
    king_player_id = GET_PLAYER_ID_BY_NAME(king_player_name)
    player_id = GET_PLAYER_ID_BY_NAME(player_name)
    num_players = num_players

    player_role = player_id + num_players - king_player_id
    if player_role >= num_players:
        player_role -= num_players
    return player_role


def get_all_historical_action(infoset):
    all_historical_action_list = list()

    if infoset is not None:
        game_init_value = infoset.game_init_value
        call_min_value = infoset.call_min_value
        all_round_exist_list = [round_num for round_num in infoset.all_round_player_action_value_dict.keys()]
        all_round_exist_list.sort()
        for round_idx, current_round in enumerate(all_round_exist_list):
            player_action_value_list = infoset.all_round_player_action_value_dict[current_round]

            for player_name, action, value_pull, value_left in player_action_value_list:
                player_init_value = infoset.player_value_init_dict[player_name]
                player_role = get_player_role(player_name, infoset.button_player_name, infoset.num_players)

                value_start = value_pull + value_left

                value_start_to_pool = value_start / game_init_value
                value_start_to_player = value_start / player_init_value
                # todo：参照player修改这块
                call_min_value_to_player_value_left = call_min_value / value_start if value_start > 0 else 0.
                value_pull_to_pool = value_pull / game_init_value
                value_pull_to_player = value_pull / player_init_value
                value_pull_to_start = value_pull / value_start
                all_historical_action_list.append((
                    round_idx, player_role, action.value,
                    value_start_to_pool, value_start_to_player,
                    call_min_value_to_player_value_left,
                    value_pull_to_pool, value_pull_to_player, value_pull_to_start
                ))
        assert len(all_historical_action_list) > 0, ValueError(f'len(all_historical_action_list) must > 0')
    return all_historical_action_list



