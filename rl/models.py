"""
This file includes the torch models. We wrap the three
models into one class for convenience.
"""

import math
import torch
from torch import nn
from env.constants import *
from utils.workflow_utils import *


class EnvEmbedding(nn.Module):
    def __init__(self, embedding_dim, num_output_class, historical_action_per_round, num_bins, do_position_embedding=False, positional_embedding_dim=128, embedding_sequence_len=-1, num_starters=1, num_acting_player_fields=9, num_other_player_fields=3, num_append_segments=0, device='cpu'):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_output_class = num_output_class
        self.historical_action_per_round = historical_action_per_round
        self.do_position_embedding = do_position_embedding
        self.positional_embedding_dim = positional_embedding_dim
        self.embedding_sequence_len = embedding_sequence_len
        self.num_starters = num_starters
        self.num_bins = num_bins
        self.num_acting_player_fields = num_acting_player_fields
        self.num_other_player_fields = num_other_player_fields
        self.device = device

        assert self.embedding_dim > 0 and self.historical_action_per_round > 0, ValueError(f'embedding_dim({self.embedding_dim}) and historical_action_per_round({self.historical_action_per_round}) must greater than 0.')

        # starters
        # all 4 rounds: pre-flop, flop, turn, river
        # all players + `not a player`
        # figures (all figures + unknown cards (opponents' card in MCTS simulation) + not dealt cards)
        # decors (all decors + unknown cards (opponents' card in MCTS simulation) + not dealt cards)
        # player status
        # bins of acting player status
        # bins of other player status
        # len(bins) + 1 means an extra bin for `empty`
        num_action_feature_bins = get_num_feature_bins_v2()
        field_dim_list = [(1, 4),
                          (1, MAX_PLAYER_NUMBER + 1),
                          (7, len(CardFigure) + 2),
                          (7, len(CardDecor) + 2),
                          (1, len(CUTTER_DEFAULT_LIST)),
                          (1, len(CUTTER_DEFAULT_LIST)),
                          (1, len(CUTTER_DEFAULT_LIST)),

                          # 行为前
                          (1, len(CUTTER_DEFAULT_LIST)),    # player_history_bet_to_pot_cutter
                          (1, len(CUTTER_BINS_LIST)),   # player_history_bet_to_assets_cutter_list
                          (1, len(CUTTER_BINS_LIST)),   # player_history_bet_to_assets_cutter_list
                          (1, len(CUTTER_SELF_BINS_LIST)),  # player_history_bet_to_assets_cutter_list
                          (1, len(CUTTER_BINS_LIST)),   # pot_to_assets_cutter_list
                          (1, len(CUTTER_BINS_LIST)),   # pot_to_assets_cutter_list
                          (1, len(CUTTER_SELF_BINS_LIST)),  # pot_to_assets_cutter_list

                          # 行为本身
                          (num_action_feature_bins, len(CUTTER_SELF_DEFAULT_LIST) + 1),     # action_value_to_pot_cutter
                          (num_action_feature_bins, len(CUTTER_BINS_LIST) + 1),     # action_value_to_assets_cutter_list
                          (num_action_feature_bins, len(CUTTER_BINS_LIST) + 1),     # action_value_to_assets_cutter_list
                          (num_action_feature_bins, len(CUTTER_BINS_LIST) + 1),     # action_value_to_assets_cutter_list

                          (MAX_PLAYER_NUMBER - 1, len(PlayerStatus)),
                          (MAX_PLAYER_NUMBER - 1, num_bins),
                          (MAX_PLAYER_NUMBER - 1, num_bins + 1),

                          # 历史行为分桶, 4个回合，historical_action_per_round个历史行为，MAX_PLAYER_NUMBER个玩家
                          (4 * historical_action_per_round * MAX_PLAYER_NUMBER, num_output_class + 1),
                          ]
        self.starter_idx_array = nn.Parameter(torch.arange(0, num_starters, dtype=torch.int32), requires_grad=False)
        current_start_idx = num_starters
        field_start_idx_list = list()
        for num_fields, num_dims in field_dim_list:
            for _ in range(num_fields):
                field_start_idx_list.append(current_start_idx)
            current_start_idx += num_dims
        self.field_start_idx_array = nn.Parameter(torch.tensor(field_start_idx_list, dtype=torch.int32), requires_grad=False)
        self.field_embedding = nn.Embedding(num_embeddings=num_starters + sum(item[1] for item in field_dim_list), embedding_dim=embedding_dim)

        # card_segments = [0, 0, 1, 1, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4]
        # num_segment_diff_by_card = len(card_segments) * 2 - max(card_segments) - 1

        if self.do_position_embedding and self.positional_embedding_dim > 0 and self.embedding_sequence_len > 0 and self.historical_action_per_round > 0:
            # 人工编码带有位置和段信息的embedding认为没有大意义
            # self.position_embedding = nn.Embedding(num_embeddings=num_starters + self.embedding_sequence_len - len(card_segments), embedding_dim=self.positional_embedding_dim)
            # position_embedding_idx_1 = torch.arange(num_starters + 2, dtype=torch.int64)
            # position_embedding_idx_2 = torch.arange(len(card_segments)).unsqueeze(1).repeat(2, 1).reshape(2 * len(card_segments)) + num_starters + 2
            # position_embedding_idx_3 = torch.arange(self.embedding_sequence_len - 2 - len(card_segments) * 2, dtype=torch.int64) + num_starters + 2 + len(card_segments)
            # self.position_embedding_idx = nn.Parameter(torch.cat([position_embedding_idx_1, position_embedding_idx_2, position_embedding_idx_3], dim=0).unsqueeze(0), requires_grad=False)
            #
            # self.segment_embedding = nn.Embedding(num_embeddings=num_starters + self.embedding_sequence_len - num_segment_diff_by_card - num_acting_player_fields + 1 - (MAX_PLAYER_NUMBER - 1) * (num_other_player_fields - 1), embedding_dim=self.positional_embedding_dim)
            # # 此处的segment组成：starters, round, role, figure * 7 * 2, decor * 7 * 2, player_status(num_player_fields) * num_player, (append_segments)
            # segment_embedding_idx_ordinary_1 = torch.arange(num_starters + 2)
            # segment_embedding_idx_ordinary_2 = torch.tensor(card_segments) + num_starters + 2
            # segment_embedding_idx_ordinary_3 = torch.tensor(card_segments) + num_starters + 2
            # segment_embedding_idx_acting_players = torch.ones(num_acting_player_fields, dtype=torch.int64) * (num_starters + self.embedding_sequence_len - num_segment_diff_by_card - num_acting_player_fields - (MAX_PLAYER_NUMBER - 1) * num_other_player_fields - num_append_segments)
            # segment_embedding_idx_other_players = torch.arange(num_starters + self.embedding_sequence_len - num_segment_diff_by_card - num_acting_player_fields + 1 - (MAX_PLAYER_NUMBER - 1) * num_other_player_fields - num_append_segments, num_starters + self.embedding_sequence_len - num_segment_diff_by_card - num_acting_player_fields + 1 - (MAX_PLAYER_NUMBER - 1) * (num_other_player_fields - 1) - num_append_segments).repeat(num_other_player_fields)
            # segment_embedding_idx_list = [segment_embedding_idx_ordinary_1, segment_embedding_idx_ordinary_2, segment_embedding_idx_ordinary_3, segment_embedding_idx_acting_players, segment_embedding_idx_other_players]
            # if num_append_segments > 0:
            #     segment_embedding_idx_list.append(torch.arange(num_starters + self.embedding_sequence_len - num_segment_diff_by_card - num_acting_player_fields + 1 - (MAX_PLAYER_NUMBER - 1) * (num_other_player_fields - 1) - num_append_segments, num_starters + self.embedding_sequence_len - num_segment_diff_by_card - num_acting_player_fields + 1 - (MAX_PLAYER_NUMBER - 1) * (num_other_player_fields - 1)))
            # self.segment_embedding_idx = nn.Parameter(torch.cat(segment_embedding_idx_list, dim=0).unsqueeze(0), requires_grad=False)
            #
            # # 加入card_embedding是因为segment_embedding无法编码牌的点数和花色的对应关系
            # self.card_embedding = nn.Embedding(num_embeddings=len(card_segments) + 1, embedding_dim=self.positional_embedding_dim)
            # card_embedding_idx_1 = torch.zeros(num_starters + 2, dtype=torch.int64)
            # card_embedding_idx_2 = torch.arange(2).unsqueeze(1).repeat(1, len(card_segments)).reshape(2 * len(card_segments)) + 1
            # card_embedding_idx_3 = torch.zeros(self.embedding_sequence_len - 2 - len(card_segments) * 2, dtype=torch.int64)
            # self.card_embedding_idx = nn.Parameter(torch.cat([card_embedding_idx_1, card_embedding_idx_2, card_embedding_idx_3], dim=0).unsqueeze(0), requires_grad=False)

            self.position_embedding = nn.Embedding(num_embeddings=num_starters + self.embedding_sequence_len, embedding_dim=self.positional_embedding_dim)
            self.position_embedding_idx = nn.Parameter(torch.arange(num_starters + self.embedding_sequence_len, dtype=torch.int32).unsqueeze(0), requires_grad=False)

    def forward(self, x):
        batch_size = x.shape[0]

        item_idx_modified_array = x + self.field_start_idx_array
        if self.num_starters > 0:
            game_status_tensor = torch.concat((self.starter_idx_array.unsqueeze(0).repeat(batch_size, 1), item_idx_modified_array), dim=1)
        else:
            game_status_tensor = item_idx_modified_array
        game_status_embedding = self.field_embedding(game_status_tensor)

        if self.do_position_embedding and self.num_starters + self.embedding_sequence_len > 0:
            position_embedding = self.position_embedding(self.position_embedding_idx.repeat(batch_size, 1))
            return game_status_embedding, position_embedding
        else:
            return game_status_embedding


class DenseActorModel(nn.Module):
    def __init__(self, embedding_dim=512, device='cpu'):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.device = device

        self.env_embedding = EnvEmbedding(embedding_dim, device=device)
        self.lstm = nn.LSTM(embedding_dim * 9, embedding_dim, batch_first=True)

        self.dense1 = nn.Linear(2 + 2 * 7 + 5 * MAX_PLAYER_NUMBER + 1, embedding_dim)
        self.relu1 = torch.nn.ReLU(inplace=False)
        self.dense2 = nn.Linear(embedding_dim, embedding_dim)
        self.relu2 = torch.nn.ReLU(inplace=False)
        self.dense3 = nn.Linear(embedding_dim, int(embedding_dim / 4))
        self.relu3 = torch.nn.ReLU(inplace=False)
        self.bn3 = nn.BatchNorm1d(embedding_dim)
        self.dense4 = nn.Linear(int(embedding_dim / 4), int(embedding_dim / 4))
        self.relu4 = torch.nn.ReLU(inplace=False)
        self.dense5 = nn.Linear(int(embedding_dim / 4), int(embedding_dim / 16))
        self.relu5 = torch.nn.ReLU(inplace=False)
        self.bn5 = nn.BatchNorm1d(embedding_dim)
        self.dense6 = nn.Linear(int(embedding_dim / 16), int(embedding_dim / 16))
        self.relu6 = torch.nn.ReLU(inplace=False)
        self.dense7 = nn.Linear(int(embedding_dim / 16), 1)
        self.relu7 = torch.nn.ReLU(inplace=False)

        self.action_dense1 = nn.Linear(embedding_dim, embedding_dim)
        self.action_relu1 = torch.nn.ReLU(inplace=False)
        self.action_dense2 = nn.Linear(embedding_dim, embedding_dim)
        self.action_relu2 = torch.nn.ReLU(inplace=False)
        self.action_dense3 = nn.Linear(embedding_dim, 4, bias=False)

        self.value_dense1 = nn.Linear(embedding_dim, embedding_dim)
        self.value_relu1 = torch.nn.ReLU(inplace=False)
        self.value_dense2 = nn.Linear(embedding_dim, embedding_dim)
        self.value_relu2 = torch.nn.ReLU(inplace=False)
        self.value_dense3 = nn.Linear(embedding_dim, 1, bias=False)

    def forward(self, x):
        game_status_embedding, historical_actions_embedding, historical_action_len_embedding = self.env_embedding(x)
        lstm_out, (h_n, _) = self.lstm(historical_actions_embedding)
        lstm_out = lstm_out[torch.arange(0, len(x), dtype=torch.long), historical_action_len_embedding, :].unsqueeze(1)
        x = torch.cat([game_status_embedding, lstm_out], dim=-2).transpose(-2, -1)
        x = self.dense1(x)
        x = self.relu1(x)
        x = self.dense2(x)
        x = self.relu2(x)
        x = self.dense3(x)
        x = self.relu3(x)
        x = self.bn3(x)
        x = self.dense4(x)
        x = self.relu4(x)
        x = self.dense5(x)
        x = self.relu5(x)
        x = self.bn5(x)
        x = self.dense6(x)
        x = self.relu6(x)
        x = self.dense7(x)
        x = self.relu7(x)
        x = x.squeeze(-1)

        action = self.action_dense1(x)
        action = self.action_relu1(action)
        action = self.action_dense2(action)
        action = self.action_relu2(action)
        action = self.action_dense3(action)
        action = torch.softmax(action, dim=1)

        value = self.value_dense1(x)
        value = self.value_relu1(value)
        value = self.value_dense2(value)
        value = self.value_relu2(value)
        value = self.value_dense3(value)
        value = value.squeeze(-1)
        value = torch.sigmoid(value)

        action = torch.argmax(action, dim=1)

        return action, value


class DenseCriticalModel(nn.Module):
    def __init__(self, embedding_dim=512, device='cpu'):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.device = device

        self.env_embedding = EnvEmbedding(embedding_dim, device=device)
        self.action_embedding = self.env_embedding.player_action
        self.action_value_embedding = nn.Embedding(num_embeddings=1, embedding_dim=embedding_dim)
        self.action_value_embedding_idx = nn.Parameter(torch.arange(1), requires_grad=False)

        self.lstm = nn.LSTM(embedding_dim * 9, embedding_dim, batch_first=True)

        self.dense1 = nn.Linear(2 + 2 * 7 + 5 * MAX_PLAYER_NUMBER + 3, embedding_dim)
        self.relu1 = torch.nn.ReLU(inplace=False)
        self.dense2 = nn.Linear(embedding_dim, embedding_dim)
        self.relu2 = torch.nn.ReLU(inplace=False)
        self.dense3 = nn.Linear(embedding_dim, int(embedding_dim / 4))
        self.relu3 = torch.nn.ReLU(inplace=False)
        self.bn3 = nn.BatchNorm1d(embedding_dim)
        self.dense4 = nn.Linear(int(embedding_dim / 4), int(embedding_dim / 4))
        self.relu4 = torch.nn.ReLU(inplace=False)
        self.dense5 = nn.Linear(int(embedding_dim / 4), int(embedding_dim / 16))
        self.relu5 = torch.nn.ReLU(inplace=False)
        self.bn5 = nn.BatchNorm1d(embedding_dim)
        self.dense6 = nn.Linear(int(embedding_dim / 16), int(embedding_dim / 16))
        self.relu6 = torch.nn.ReLU(inplace=False)
        self.dense7 = nn.Linear(int(embedding_dim / 16), 1)
        self.relu7 = torch.nn.ReLU(inplace=False)

        self.value_dense1 = nn.Linear(embedding_dim, embedding_dim)
        self.value_relu1 = torch.nn.ReLU(inplace=False)
        self.value_dense2 = nn.Linear(embedding_dim, embedding_dim)
        self.value_relu2 = torch.nn.ReLU(inplace=False)
        self.value_dense3 = nn.Linear(embedding_dim, 1, bias=False)

    def forward(self, x, action):
        game_status_embedding, historical_actions_embedding, historical_action_len_embedding = self.env_embedding(x)
        player_action_embedding = self.action_embedding(action[0]).unsqueeze(1).to(self.device)
        player_action_value_embedding = torch.multiply(self.action_value_embedding(self.action_value_embedding_idx.repeat(len(action[1]))), action[1].unsqueeze(1).to(self.device)).unsqueeze(1)

        lstm_out, (h_n, _) = self.lstm(historical_actions_embedding)
        lstm_out = lstm_out[torch.arange(0, len(x), dtype=torch.long), historical_action_len_embedding, :].unsqueeze(1)

        x = torch.cat([game_status_embedding, lstm_out, player_action_embedding, player_action_value_embedding], dim=-2).transpose(-2, -1)
        x = self.dense1(x)
        x = self.relu1(x)
        x = self.dense2(x)
        x = self.relu2(x)
        x = self.dense3(x)
        x = self.relu3(x)
        x = self.bn3(x)
        x = self.dense4(x)
        x = self.relu4(x)
        x = self.dense5(x)
        x = self.relu5(x)
        x = self.bn5(x)
        x = self.dense6(x)
        x = self.relu6(x)
        x = self.dense7(x)
        x = self.relu7(x)
        x = x.squeeze(-1)

        value = self.value_dense1(x)
        value = self.value_relu1(value)
        value = self.value_dense2(value)
        value = self.value_relu2(value)
        value = self.value_dense3(value)
        value = value.squeeze(-1)

        return value


class TransformerActorModel(nn.Module):
    def __init__(self,
                 num_bins,
                 embedding_dim=512,
                 positional_embedding_dim=128,
                 num_layers=6,
                 historical_action_per_round=6,
                 num_player_fields=10,
                 device='cpu'
                 ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.device = device

        # round, role, figure * 7 * 2, decor * 7 * 2, player_status(num_player_fields) * num_player
        embedding_sequence_len = 2 + 4 * 7 + num_player_fields * MAX_PLAYER_NUMBER

        self.env_embedding = EnvEmbedding(embedding_dim,
                                          historical_action_per_round=historical_action_per_round,
                                          num_bins=num_bins,
                                          do_position_embedding=True,
                                          positional_embedding_dim=positional_embedding_dim,
                                          embedding_sequence_len=embedding_sequence_len,
                                          num_starters=2,
                                          num_player_fields=num_player_fields,
                                          device=self.device)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim + positional_embedding_dim, nhead=8, batch_first=True, device=self.device)
        encoder_norm = nn.LayerNorm(normalized_shape=embedding_dim + positional_embedding_dim, device=self.device)
        self.transform_encoder = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)

        self.action_dense = nn.Linear(embedding_dim + positional_embedding_dim, 4)
        self.action_value_dense = nn.Linear(embedding_dim + positional_embedding_dim, 1)

    def forward(self, x):
        game_status_embedding, position_segment_embedding = self.env_embedding(x)

        x = torch.cat([game_status_embedding, position_segment_embedding], dim=2)
        x = self.transform_encoder(x)

        action_x = x[:, 0, :]
        action_value_x = x[:, 1, :]

        action = self.action_dense(action_x)
        action = torch.softmax(action, dim=1)
        action = torch.argmax(action, dim=1)

        action_value = self.action_value_dense(action_value_x).squeeze(-1)
        action_value = torch.sigmoid(action_value)

        return action, action_value


class TransformerCriticalModel(nn.Module):
    def __init__(self,
                 num_bins,
                 embedding_dim=512,
                 positional_embedding_dim=128,
                 num_layers=6,
                 historical_action_per_round=6,
                 num_player_fields=10,
                 device='cpu'
                 ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.positional_embedding_dim = positional_embedding_dim
        self.historical_action_per_round = historical_action_per_round
        self.device = device

        # round, role, figure * 7 * 2, decor * 7 * 2, player_status(num_player_fields) * num_player, action, action_value
        embedding_sequence_len = 2 + 4 * 7 + num_player_fields * MAX_PLAYER_NUMBER + 2

        self.env_embedding = EnvEmbedding(embedding_dim,
                                          historical_action_per_round=historical_action_per_round,
                                          num_bins=num_bins,
                                          do_position_embedding=True,
                                          positional_embedding_dim=positional_embedding_dim,
                                          embedding_sequence_len=embedding_sequence_len,
                                          num_starters=2,
                                          num_append_segments=2,
                                          num_player_fields=num_player_fields,
                                          device=self.device)
        self.action_embedding = self.env_embedding.player_action
        self.action_value_embedding = nn.Embedding(num_embeddings=1, embedding_dim=embedding_dim)
        self.action_value_embedding_idx = nn.Parameter(torch.arange(1), requires_grad=False)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim + positional_embedding_dim, nhead=8, batch_first=True, device=self.device)
        encoder_norm = nn.LayerNorm(normalized_shape=embedding_dim + positional_embedding_dim, device=self.device)
        self.transform_encoder = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)

        # critical网络的训练目标是value最大化，loss出现负值是可以接受的
        self.reward_value_dense = nn.Linear(embedding_dim + positional_embedding_dim, 1)
        self.player_result_value_dense = nn.Linear(embedding_dim + positional_embedding_dim, 1)

    def forward(self, x, action):
        game_status_embedding, position_segment_embedding = self.env_embedding(x)

        player_action_embedding = self.action_embedding(action[0]).unsqueeze(1).to(self.device)
        player_action_value_embedding = torch.multiply(self.action_value_embedding(self.action_value_embedding_idx.repeat(len(action[1]))), action[1].unsqueeze(1).to(self.device)).unsqueeze(1)

        x = torch.cat([game_status_embedding, player_action_embedding, player_action_value_embedding], dim=1)
        transformer_input = torch.cat([x, position_segment_embedding], dim=2)
        transformer_output = self.transform_encoder(transformer_input)

        reward_value = self.reward_value_dense(transformer_output[:, 0, :])
        reward_value = reward_value.squeeze(-1)

        player_result_value = self.player_result_value_dense(transformer_output[:, 1, :])
        player_result_value = player_result_value.squeeze(-1)

        return reward_value, player_result_value


class TransformerAlphaGoZeroModel(nn.Module):
    def __init__(self,
                 num_bins,
                 num_winning_prob_bins,
                 num_output_class,
                 embedding_dim=512,
                 positional_embedding_dim=128,
                 num_layers=6,
                 transformer_head_dim=64,
                 historical_action_per_round=6,
                 num_acting_player_fields=9,
                 num_other_player_fields=3,
                 device='cpu'
                 ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.positional_embedding_dim = positional_embedding_dim
        self.historical_action_per_round = historical_action_per_round
        self.device = device

        # to normalize embedding for transformer
        self.actual_embedding_dim = int(embedding_dim + positional_embedding_dim)
        self.sqrt_of_actual_embedding_dim = math.sqrt(self.actual_embedding_dim)

        # round, role, figure * 7, decor * 7, acting_player_features, other_player_features * num_other_players, round_number * historical_action_per_round * all_player_number
        embedding_sequence_len = 2 + 2 * 7 + num_acting_player_fields + num_other_player_fields * (MAX_PLAYER_NUMBER - 1) + 4 * historical_action_per_round * MAX_PLAYER_NUMBER

        self.env_embedding = EnvEmbedding(embedding_dim,
                                          num_output_class=num_output_class,
                                          historical_action_per_round=historical_action_per_round,
                                          num_bins=num_bins,
                                          do_position_embedding=True,
                                          positional_embedding_dim=positional_embedding_dim,
                                          embedding_sequence_len=embedding_sequence_len,
                                          num_starters=5,
                                          num_acting_player_fields=num_acting_player_fields,
                                          num_other_player_fields=num_other_player_fields,
                                          device=self.device)

        assert self.actual_embedding_dim % transformer_head_dim == 0, f'embedding_dim({embedding_dim}) + positional_embedding_dim({positional_embedding_dim}) * 3 must be divisible by transformer_head_dim({transformer_head_dim}).'
        num_transformer_head = self.actual_embedding_dim // transformer_head_dim
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.actual_embedding_dim, nhead=num_transformer_head, batch_first=True)
        self.transform_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.card_result_value_logits_sigmoid = torch.nn.Sigmoid()

        # 预测action的概率分布
        self.action_prob_dense = nn.Linear(self.actual_embedding_dim, num_output_class, bias=True)
        # 预测action的Q值分布
        self.reward_value_dense = nn.Linear(self.actual_embedding_dim, 1, bias=True)
        # 预测'客观胜负'
        self.card_result_value_dense = nn.Linear(self.actual_embedding_dim, 1, bias=True)
        # 预测玩家'客观牌力范围'
        self.player_winning_prob_dense = nn.Linear(self.actual_embedding_dim, num_winning_prob_bins, bias=True)
        # 预测对手'客观牌力范围'
        # todo：当前只支持两个玩家
        self.opponent_winning_prob_dense = nn.Linear(self.actual_embedding_dim, num_winning_prob_bins, bias=True)

    def forward(self, x):
        game_status_embedding, position_segment_embedding = self.env_embedding(x)

        x = torch.concat([game_status_embedding, position_segment_embedding], dim=2)
        # to normalize embedding
        x = x * self.sqrt_of_actual_embedding_dim
        x = self.transform_encoder(x)
        # x = x.reshape(x.shape[0], -1)

        action_prob_logits = self.action_prob_dense(x[:, 0, :])
        # action_prob = self.action_logits_softmax(action_prob_logits)

        reward_value_logits = self.reward_value_dense(x[:, 1, :]).squeeze(1)

        card_result_value_logits = self.card_result_value_logits_sigmoid(self.card_result_value_dense(x[:, 2, :])).squeeze(1)

        player_winning_prob_logits = self.player_winning_prob_dense(x[:, 3, :])

        opponent_winning_prob_logits = self.opponent_winning_prob_dense(x[:, 4, :])

        return action_prob_logits, reward_value_logits, card_result_value_logits, player_winning_prob_logits, opponent_winning_prob_logits
