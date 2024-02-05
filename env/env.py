import logging
import random
import numpy as np
from tools.biner import *
from .cards import *
from .game import GameEnv, deck
from utils.workflow_utils import map_action_bin_to_actual_action_and_value_v2

class Env:
    """
    Doudizhu multi-agent wrapper
    """
    def __init__(self, winning_probability_generating_task_queue, num_bins, num_players: int = MAX_PLAYER_NUMBER, init_value: int = 100_000, small_blind=25, big_blind=50, num_player_fields=3, game_env=None, ignore_all_async_tasks=False, settle_automatically=True):
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
        self.settle_automatically = settle_automatically

        self.num_players = num_players
        assert num_players <= MAX_PLAYER_NUMBER, f'There are up to {MAX_PLAYER_NUMBER} players in the game, received {num_players}'

        self.init_value = init_value
        self.small_blind = small_blind
        self.big_blind = big_blind

        self.num_player_fields = num_player_fields

        # Initialize the internal environment
        if game_env is None:
            self._env = GameEnv(self.num_players, self.init_value, small_blind=self.small_blind, big_blind=self.big_blind, settle_automatically=settle_automatically)
        else:
            self._env = game_env

        # 使用round_num + player_name作为key过滤任务，减少无用计算
        self.reward_cal_task_set = set()

        # 初始化连续特征的分桶器
        self.player_init_value_to_game_init_cutter = CutByThreshold(np.array(CUTTER_DEFAULT_LIST) * MAX_PLAYER_NUMBER)
        self.player_value_left_to_game_init_cutter = CutByThreshold(np.array(CUTTER_DEFAULT_LIST) * MAX_PLAYER_NUMBER)
        self.player_value_left_to_player_init_cutter = CutByThreshold(np.array(CUTTER_DEFAULT_LIST))

        # 行为前
        self.player_history_bet_to_pot_cutter = CutByThreshold(np.array(CUTTER_DEFAULT_LIST))
        self.player_history_bet_to_assets_cutter_list = [
            CutByThreshold(np.array(CUTTER_BINS_LIST) * MAX_PLAYER_NUMBER),
            CutByThreshold(np.array(CUTTER_BINS_LIST)),
            CutByThreshold(np.array(CUTTER_SELF_BINS_LIST)),
        ]
        self.pot_to_assets_cutter_list = [
            CutByThreshold(np.array(CUTTER_BINS_LIST) * MAX_PLAYER_NUMBER),
            CutByThreshold(np.array(CUTTER_BINS_LIST) * MAX_PLAYER_NUMBER),
            CutByThreshold(np.array(CUTTER_SELF_BINS_LIST)),
        ]

        # 行为本身
        self.action_value_to_pot_cutter = CutByThreshold(np.array(CUTTER_SELF_DEFAULT_LIST), include_invalid_bin=True)
        self.action_value_to_assets_cutter_list = [
            CutByThreshold(np.array(CUTTER_BINS_LIST) * MAX_PLAYER_NUMBER, include_invalid_bin=True),
            CutByThreshold(np.array(CUTTER_BINS_LIST), include_invalid_bin=True),
            CutByThreshold(np.array(CUTTER_BINS_LIST), include_invalid_bin=True),
        ]

        # # 行为后
        # self.after_action_player_history_bet_to_pots_cutter = CutByThreshold(np.array(CUTTER_DEFAULT_LIST), include_invalid_bin=True)
        # self.after_action_player_history_bet_to_assets_cutter_list = [
        #     CutByThreshold(np.array(CUTTER_BINS_LIST) * MAX_PLAYER_NUMBER, include_invalid_bin=True),
        #     CutByThreshold(np.array(CUTTER_BINS_LIST), include_invalid_bin=True),
        #     CutByThreshold(np.array(CUTTER_SELF_BINS_LIST), include_invalid_bin=True),
        # ]
        # self.after_action_pot_to_assets_cutter_list = [
        #     CutByThreshold(np.array(CUTTER_BINS_LIST) * MAX_PLAYER_NUMBER, include_invalid_bin=True),
        #     CutByThreshold(np.array(CUTTER_BINS_LIST) * MAX_PLAYER_NUMBER, include_invalid_bin=True),
        #     CutByThreshold(np.array(CUTTER_SELF_BINS_LIST), include_invalid_bin=True),
        # ]

        # 玩家间特征
        self.player_value_left_to_init_value_cutter = CutByThreshold(np.array(CUTTER_DEFAULT_LIST))
        # 以下，其他玩家财力大于当前玩家越多，当前玩家下大价值风险越大，则要对小于1和大于1的部分都有细分刻画，且包含一个远大于的部分
        self.player_init_value_to_acting_player_init_value_cutter = CutByThreshold(np.array(CUTTER_DEFAULT_LIST + [1.]) * MAX_PLAYER_NUMBER)

        self.game_id = None

    def reset(self, game_id, seed=None, cards_dict=None):
        """
        Every time reset is called, the environment
        will be re-initialized with a new deck of cards.
        This function is usually called when a game is over.
        """
        logging.debug(f'env reset')
        self.game_id = game_id

        # reset environment
        self._env.reset(seed=seed, cards_dict=cards_dict)

        logging.debug([f'{player_name}:{int(player.value_left)},{player.status.name}' for player_name, player in self._env.players.items()])

        infoset = self._game_infoset

        return self.get_obs(infoset)

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

        if self.game_over:
            if self.settle_automatically:
                done = True
                reward = self._get_final_reward()
                logging.debug(f'reward={reward}')
                # obs = self.get_obs(self._env.info_sets, is_last_round=True)
                obs = None

                if not self.ignore_all_async_tasks:
                    self.reward_cal_task_set.clear()
            else:
                done = True
                reward = None
                obs = None
        else:
            done = False
            # reward = self._get_reward(game_id, step_id, acted_player_name, self.reward_cal_task_dict)
            reward = None
            obs = self.get_obs(self._game_infoset)
        return obs, reward, done, info

    def new_random(self):
        # new_random只会在MCTS模拟中用到，所以设置settle_automatically为True
        return Env(winning_probability_generating_task_queue=None,
                   num_bins=self.num_bins,
                   num_players=self.num_players,
                   init_value=self.init_value,
                   small_blind=self.small_blind,
                   big_blind=self.big_blind,
                   game_env=self._env.new_random(),
                   ignore_all_async_tasks=True,
                   settle_automatically=True)

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

    # def _get_reward_value(self):
    #     """
    #     This function is called in the end of each
    #     game. It returns either 1/-1 for win/loss,
    #     or ADP, i.e., every bomb will double the score.
    #     """
    #     player_role_reward_value = dict()
    #     for player_name, init_value in self._env.player_init_value_dict.items():
    #         player = self._env.players[player_name]
    #
    #         player_role = get_player_role(player_name, self._env.button_player_name, self._env.num_players)
    #         player_role_reward_value[player_role] = (player.value_left - init_value) / init_value
    #     return player_role_reward_value

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
    def acting_player_name(self):
        """
        The player that is active. It can be landlord,
        landlod_down, or landlord_up.
        """
        return self._env.acting_player_name

    @property
    def game_over(self):
        """ Returns a Boolean
        """
        return self._env.game_over

    def _map_obs_idx_to_model_index(self, current_round, current_player_role, cards, acting_player_status_list, other_player_status_list):
        sorted_item_list = list()
        sorted_item_list.append(current_round)
        sorted_item_list.append(current_player_role)

        all_figure, all_decor = list(), list()
        for card_representation_list in cards:
            for card_representation in card_representation_list:
                all_figure.append(card_representation[0])
                all_decor.append(card_representation[1])
        # for card_representation in sorted_cards:
        #     all_figure.append(card_representation[0])
        #     all_decor.append(card_representation[1])
        sorted_item_list.extend(all_figure)
        sorted_item_list.extend(all_decor)

        sorted_item_list.extend(acting_player_status_list)

        for i in range(self.num_player_fields):
            for j in range(MAX_PLAYER_NUMBER - 1):
                sorted_item_list.append(other_player_status_list[j][i])

        return np.array(sorted_item_list, dtype=np.int32)

    def get_valid_action_info(self):
        return self._env.get_valid_action_info()

    def get_valid_action_info_v2(self):
        return self._env.get_valid_action_info_v2()

    def get_obs(self, infoset):
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
        current_round = infoset.current_round
        # 注意这用相对位置，因为button位在局和局间变化，此处要刻画在该局游戏中玩家位置的强弱
        acting_player_id = get_relative_player_position(infoset.player_name, infoset.button_player_name, infoset.num_players)

        hand_cards = get_card_representations(infoset.player_hand_cards, 2)
        flop_cards = get_card_representations(infoset.flop_cards, 3)
        turn_cards = get_card_representations(infoset.turn_cards, 1)
        river_cards = get_card_representations(infoset.river_cards, 1)

        # sorted_cards = sort_card_representations(hand_cards, flop_cards, turn_cards, river_cards)

        acting_player_status_list, other_player_status_list = self.get_all_player_current_status(infoset)

        # all_historical_player_action = get_all_historical_action(infoset)
        # 干掉sorted_cards，疑似重复信息字段，测试模型是否可以学到规则
        # return self._map_obs_idx_to_model_index(current_round, acting_player_id,
        #         [hand_cards, flop_cards, turn_cards, river_cards],
        #         sorted_cards, acting_player_status_list, other_player_status_list)
        return self._map_obs_idx_to_model_index(current_round, acting_player_id,
                [hand_cards, flop_cards, turn_cards, river_cards],
                acting_player_status_list, other_player_status_list)

    def get_all_player_current_status(self, infoset):
        assert infoset is not None, "The infoset should not be None."

        all_player_current_status_list = [None] * (infoset.num_players - 1)

        game_init_value = infoset.game_init_value
        # 当前游戏整体下注价值
        pot_value = infoset.pot_value
        # 当前玩家本轮所有bin的行为价值（预估）
        bin_bet_value_list = infoset.bin_bet_value_list
        num_bins = len(bin_bet_value_list)

        acting_player_init_value = infoset.player_value_init_dict[infoset.player_name]
        _, acting_player_left_value, acting_player_history_bet_value = infoset.player_status_value_left_bet_dict[infoset.player_name]

        # 资产包括：game_init_value（游戏绝对价值），acting_player_init_value（玩家绝对），acting_player_left_value（玩家相对）
        all_asset_value_list = [game_init_value, acting_player_init_value, acting_player_left_value]
        # 行为后总下注（预估）
        after_action_history_bet_value_list = [acting_player_history_bet_value + bet_value for bet_value in bin_bet_value_list]
        # 行为后池价值（预估）
        after_action_pot_value_list = [pot_value + bet_value for bet_value in bin_bet_value_list]
        # 行为后资产（预估）
        after_action_asset_value_list = [[game_init_value] * num_bins, [acting_player_init_value] * num_bins, [acting_player_left_value - bet_value for bet_value in bin_bet_value_list]]

        # 特征值开始
        # 提示局势整体的激进/保守偏好
        player_init_to_game_init = (acting_player_init_value, game_init_value)
        # 提示当前行为的绝对参照性的激进/保守偏好
        player_left_to_game_init = (acting_player_left_value, game_init_value)
        # 提示当前行为的局势相对性的激进/保守偏好
        player_left_to_player_init = (acting_player_left_value, acting_player_init_value)

        # 行为前下注-行为前池
        player_history_bet_to_pot = (acting_player_history_bet_value, pot_value)
        # 行为前下注-行为前资产
        player_history_bet_to_assets = [(acting_player_history_bet_value, asset_value) for asset_value in all_asset_value_list]
        # 行为前池-行为前资产
        pot_to_assets = [(pot_value, asset_value) for asset_value in all_asset_value_list]

        # 行为-行为前池
        action_value_to_pots = [(action_value, pot_value, action_value) for action_value in bin_bet_value_list]
        # 行为-行为前资产
        action_value_to_assets_list = [[(action_value, asset, action_value) for action_value in bin_bet_value_list] for asset in all_asset_value_list]

        # 行为后下注-行为后池
        after_action_player_history_bet_to_pots = [(after_action_history_bet_value, after_action_pot_value, action_value) for after_action_history_bet_value, after_action_pot_value, action_value in zip(after_action_history_bet_value_list, after_action_pot_value_list, bin_bet_value_list)]
        # 行为后下注-行为后资产
        after_action_player_history_bet_to_assets_list = [[(after_action_history_bet_value, after_action_asset_value, action_value) for after_action_history_bet_value, after_action_asset_value, action_value in zip(after_action_history_bet_value_list, after_action_asset_values, bin_bet_value_list)] for after_action_asset_values in after_action_asset_value_list]
        # 行为后池-行为后资产
        after_action_pot_to_assets_list = [[(after_action_pot_value, after_action_asset_value, action_value) for after_action_pot_value, after_action_asset_value, action_value in zip(after_action_pot_value_list, after_action_asset_values, bin_bet_value_list)] for after_action_asset_values in after_action_asset_value_list]
        # 特征值结束

        # 分桶
        acting_player_status = list()
        acting_player_status.append(self.player_init_value_to_game_init_cutter.cut(*player_init_to_game_init))
        acting_player_status.append(self.player_value_left_to_game_init_cutter.cut(*player_left_to_game_init))
        acting_player_status.append(self.player_value_left_to_player_init_cutter.cut(*player_left_to_player_init))

        # 行为前
        acting_player_status.append(self.player_history_bet_to_pot_cutter.cut(*player_history_bet_to_pot))
        acting_player_status.extend(player_history_bet_to_assets_cutter.cut(*player_history_bet_to_asset) for player_history_bet_to_assets_cutter, player_history_bet_to_asset in zip(self.player_history_bet_to_assets_cutter_list, player_history_bet_to_assets))
        acting_player_status.extend(pot_to_assets_cutter.cut(*pot_to_asset) for pot_to_assets_cutter, pot_to_asset in zip(self.pot_to_assets_cutter_list, pot_to_assets))

        # 行为本身
        acting_player_status.extend(self.action_value_to_pot_cutter.cut(*action_value_to_pot) for action_value_to_pot in action_value_to_pots)
        for action_value_to_assets_cutter, action_value_to_assets in zip(self.action_value_to_assets_cutter_list, action_value_to_assets_list):
            for action_value_to_asset in action_value_to_assets:
                acting_player_status.append(action_value_to_assets_cutter.cut(*action_value_to_asset))

        # # 行为后
        # acting_player_status.extend(self.after_action_player_history_bet_to_pots_cutter.cut(*after_action_player_history_bet_to_pot) for after_action_player_history_bet_to_pot in after_action_player_history_bet_to_pots)
        # for after_action_player_history_bet_to_assets_cutter, after_action_player_history_bet_to_assets in zip(self.after_action_player_history_bet_to_assets_cutter_list, after_action_player_history_bet_to_assets_list):
        #     for after_action_player_history_bet_to_asset in after_action_player_history_bet_to_assets:
        #         acting_player_status.append(after_action_player_history_bet_to_assets_cutter.cut(*after_action_player_history_bet_to_asset))
        # for after_action_pot_to_assets_cutter, after_action_pot_to_assets in zip(self.after_action_pot_to_assets_cutter_list, after_action_pot_to_assets_list):
        #     for after_action_pot_to_asset in after_action_pot_to_assets:
        #         acting_player_status.append(after_action_pot_to_assets_cutter.cut(*after_action_pot_to_asset))

        for current_player_name, player_init_value in infoset.player_value_init_dict.items():
            if current_player_name != infoset.player_name:
                player_status, player_left_value, player_bet_value = infoset.player_status_value_left_bet_dict[current_player_name]
                # 对于当前玩家的相对位置
                # todo：此处相对位置和绝对位置都有物理意义，要怎么刻画没想好
                relative_player_position = get_relative_player_position(current_player_name, infoset.button_player_name, infoset.num_players) - 1

                # 提示玩家下注比例
                player_value_left_to_init_value = (player_left_value, player_init_value)
                # 提示玩家实力对比
                player_init_value_to_acting_player_init_value = (player_init_value, acting_player_init_value)

                player_value_left_to_init_value_bin = self.player_value_left_to_init_value_cutter.cut(*player_value_left_to_init_value)
                player_init_value_to_acting_player_init_value_bin = self.player_init_value_to_acting_player_init_value_cutter.cut(*player_init_value_to_acting_player_init_value)

                all_player_current_status_list[relative_player_position] = (player_status.value,
                                                                            player_value_left_to_init_value_bin,
                                                                            player_init_value_to_acting_player_init_value_bin)
        return acting_player_status, all_player_current_status_list


class RandomEnv(Env):

    def __init__(self, winning_probability_generating_task_queue, num_bins, num_players: int = MAX_PLAYER_NUMBER, init_value: int = 100_000, small_blind=25, big_blind=50, num_player_fields=3, game_env=None, ignore_all_async_tasks=False, settle_automatically=True, action_probs=None, round_probs=None):
        Env.__init__(self, winning_probability_generating_task_queue=winning_probability_generating_task_queue, num_bins=num_bins, num_players=num_players, init_value=init_value, small_blind=small_blind, big_blind=big_blind, num_player_fields=num_player_fields, game_env=game_env, ignore_all_async_tasks=ignore_all_async_tasks, settle_automatically=settle_automatically)

        if action_probs is not None:
            self.action_probs = action_probs
        else:
            # fold会导致游戏提前结束
            # allin后跟call或allin都会导致游戏提前结束
            self.action_probs = np.asarray([0., 0.4, 0.3, 0.1, 0.05, 0.05, 0.05, 0.02, 0.02, 0.005, 0.005, 0.])

        if round_probs is not None:
            self.round_probs = round_probs
        else:
            self.round_probs = np.asarray([0.01, 0.09, 0.2, 0.3, 0.4])

        self.target_round_step_stop_probs = [0.2, 0.25, 0.333, 0.5]

    def __generate_random_step(self, action_probs, force_check=False):
        action_mask_list, action_value_or_ranges_list, acting_player_value_left, current_round_acting_player_historical_value, delta_min_value_to_raise = self.get_valid_action_info_v2()
        if force_check and not action_mask_list[1]:
            return PlayerActions.CHECK_CALL, self._env.current_round_min_value
        else:
            valid_action_probs = np.copy(action_probs)
            valid_action_probs[action_mask_list] = 0

            sum_probs = sum(valid_action_probs)
            if sum_probs == 0:
                raise ZeroDivisionError(f"all action_probs are masked due to a forced all in by other player")
            valid_action_probs /= sum_probs

            random_num = random.random()
            cumulative_prob = 0
            for i, prob in enumerate(valid_action_probs):
                cumulative_prob += prob
                if random_num <= cumulative_prob:
                    return map_action_bin_to_actual_action_and_value_v2(i, action_value_or_ranges_list, acting_player_value_left, current_round_acting_player_historical_value, delta_min_value_to_raise, self.small_blind)
            raise ValueError(f"action_probs should be a probability distribution, but={action_probs}")

    def take_step_to_round(self, max_round):
        current_round = 0
        while current_round <= max_round:
            num_bets = 0
            num_check = 0
            stop_simulation = False
            target_round_step_num = 0
            while True:
                if current_round == max_round:
                    # 最后一轮，每一步都在一个随机概率下停止模拟
                    if target_round_step_num < len(self.target_round_step_stop_probs):
                        random_num = random.random()
                        if random_num <= self.target_round_step_stop_probs[target_round_step_num]:
                            stop_simulation = True
                            break
                    target_round_step_num += 1

                if num_bets >= 2:
                    force_check = True
                else:
                    force_check = False
                action = self.__generate_random_step(self.action_probs, force_check=force_check)
                if action[0] == PlayerActions.RAISE:
                    num_bets += 1
                elif action[0] == PlayerActions.CHECK_CALL:
                    num_check += 1
                    # 最后一轮，不能连续check，不能bet后check，会导致回合结束
                    if current_round == max_round and (num_check > 1 or (num_bets > 0 and num_check > 0)):
                        stop_simulation = True
                        break

                self._env.step(action)
                if current_round != self._env.current_round:
                    break
                if self.game_over:
                    stop_simulation = True
                    break

            if stop_simulation:
                break
            else:
                current_round = self._env.current_round

    def reset(self, game_id, seed=None, cards_dict=None):
        obs = Env.reset(self, game_id=game_id, seed=seed, cards_dict=cards_dict)
        random_num = random.random()
        cumulative_prob = 0
        target_round = None
        for i, prob in enumerate(self.round_probs):
            cumulative_prob += prob
            if random_num <= cumulative_prob:
                if i == 0:
                    return obs
                else:
                    target_round = i - 1
                    break
        assert target_round is not None, f'self.round_probs should be a probability distribution, but={self.round_probs}'

        while True:
            # 强制设置的概率中包含0，可能会导致self.get_valid_action_info()中mask时把所有有效数值全置为0，导致无法生成有效的action
            # 这是因为其他玩家的raise把当前玩家逼到只能allin或fold
            # 而这种情况导致无法到达指定的round，认为是无效的初始对局
            try:
                self.take_step_to_round(target_round)
                if self.game_over:
                    Env.reset(self, game_id=game_id, seed=seed, cards_dict=cards_dict)
                    continue
            except ZeroDivisionError:
                Env.reset(self, game_id=game_id, seed=seed, cards_dict=cards_dict)
                continue
            except Exception as e:
                raise e
            break

        # 注意此处游戏是初始化状态，不需要生成reward计算任务
        return self.get_obs(self._game_infoset)


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


def get_relative_player_position(player_name, base_player_name, num_players):
    base_player_id = GET_PLAYER_ID_BY_NAME(base_player_name)
    player_id = GET_PLAYER_ID_BY_NAME(player_name)

    player_role = player_id - base_player_id
    if player_role < 0:
        player_role += num_players
    return player_role
