"""AlphaGoZero network for Splendor.

Same design as rl/models.TransformerAlphaGoZeroModel + rl/AlphaGoZero.AlphaGoZero (poker):
    * every observation field is an integer looked up in one shared embedding table
      (one vocab range per field group, fields of a group share it);
    * a factorised field-position embedding (attr + owner + slot) is ADDED to the token
      embedding (no concatenation, no sqrt(d) scaling), then a pre-norm TransformerEncoder with a
      final LayerNorm; the embedding is initialised at the scale of one sublayer's residual write;
    * learnable "starter" tokens in front of the sequence carry the heads:
      starter 0 -> action probability logits, starter 1 -> value;
    * Adam + CyclicLR, SimpleMemory replay buffer, optional text dump of train data.

Differences: the value head outputs one value per seat (relative to the acting player,
4-player game), and the policy loss masks illegal actions before the softmax.
"""

import logging
import math
from threading import Lock

import numpy as np
import torch
from torch import nn

from tools.memories.simple_memory import SimpleMemory
from utils.torch_utils import PRE_NORM_FFN_WRITE_RMS, resolve_embedding_init_rms
from splendor.rl_env import (FIELD_GROUPS, VOCAB_SIZES, ATTRS, OWNERS, SLOTS, FIELD_ATTR_IDX, FIELD_OWNER_IDX,
                             FIELD_SLOT_IDX, N_ACTIONS, N_OBSERVATION, N_PLAYERS)

ILLEGAL_LOGIT = -1e4


def resolve_device(device):
    if device in (None, 'auto'):
        if torch.cuda.is_available():
            return 'cuda:0'
        if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'
    return device


class SplendorEnvEmbedding(nn.Module):
    """token embedding + factorised field position embedding, summed.

    * field_embedding: one shared table; each field's integer value is offset by its vocab
      group's start, so fields of the same group (e.g. card_id of all 24 card slots) share rows
      and different groups never collide. Rows [0, num_starters) are the starter tokens.
    * position (= field identity, the fields have no meaningful order):
          E_attr[what the field measures] + E_owner[global/market/player k] + E_slot[which entity]
      see splendor.rl_env.FIELD_LAYOUT. Starter tokens get no position: their rows are unique.
    * no sqrt(d) scaling. The summed embedding is the first write into the pre-norm residual
      stream; it is initialised with total RMS `init_rms` (default: the RMS one FFN sublayer writes
      at init, utils.torch_utils.PRE_NORM_FFN_WRITE_RMS), split evenly between content and
      position: field table std = init_rms/sqrt(2), each of the 3 position tables std = init_rms/sqrt(6).
    """

    def __init__(self, embedding_dim, num_starters, init_rms=PRE_NORM_FFN_WRITE_RMS):
        super().__init__()
        self.num_starters = num_starters

        group_offsets = dict()
        current = num_starters
        for group in ATTRS:
            group_offsets[group] = current
            current += VOCAB_SIZES[group]
        self.register_buffer('field_start_idx_array', torch.tensor([group_offsets[g] for g in FIELD_GROUPS], dtype=torch.int64))
        self.register_buffer('starter_idx_array', torch.arange(0, num_starters, dtype=torch.int64))
        self.field_embedding = nn.Embedding(num_embeddings=current, embedding_dim=embedding_dim)

        self.register_buffer('field_attr_idx', torch.tensor(FIELD_ATTR_IDX, dtype=torch.int64))
        self.register_buffer('field_owner_idx', torch.tensor(FIELD_OWNER_IDX, dtype=torch.int64))
        self.register_buffer('field_slot_idx', torch.tensor(FIELD_SLOT_IDX, dtype=torch.int64))
        self.attr_embedding = nn.Embedding(num_embeddings=len(ATTRS), embedding_dim=embedding_dim)
        self.owner_embedding = nn.Embedding(num_embeddings=len(OWNERS), embedding_dim=embedding_dim)
        self.slot_embedding = nn.Embedding(num_embeddings=len(SLOTS), embedding_dim=embedding_dim)

        if init_rms is not None:
            nn.init.normal_(self.field_embedding.weight, mean=0., std=init_rms / math.sqrt(2.))
            for embedding in (self.attr_embedding, self.owner_embedding, self.slot_embedding):
                nn.init.normal_(embedding.weight, mean=0., std=init_rms / math.sqrt(6.))

    def position_embedding(self):
        """[num_starters + N_OBSERVATION, embedding_dim]; zeros for the starter tokens."""
        field_position = (self.attr_embedding(self.field_attr_idx)
                          + self.owner_embedding(self.field_owner_idx)
                          + self.slot_embedding(self.field_slot_idx))
        starter_position = field_position.new_zeros((self.num_starters, field_position.shape[1]))
        return torch.cat((starter_position, field_position), dim=0)

    def forward(self, x):
        batch_size = x.shape[0]
        item_idx = x.long() + self.field_start_idx_array
        tokens = torch.cat((self.starter_idx_array.unsqueeze(0).expand(batch_size, -1), item_idx), dim=1)
        # [B, L, d] + [L, d] broadcast
        return self.field_embedding(tokens) + self.position_embedding()


class SplendorTransformerModel(nn.Module):
    def __init__(self, n_actions=N_ACTIONS, n_players=N_PLAYERS, embedding_dim=256,
                 num_layers=6, transformer_head_dim=64, dropout=0.1, embedding_init_rms='auto', dim_feedforward=2048):
        super().__init__()
        assert embedding_dim % transformer_head_dim == 0, \
            f'embedding_dim({embedding_dim}) must be divisible by transformer_head_dim({transformer_head_dim}).'
        self.embedding_dim = embedding_dim

        self.env_embedding = SplendorEnvEmbedding(embedding_dim, num_starters=2, init_rms=resolve_embedding_init_rms(embedding_init_rms))
        # pre-norm（x = x + f(LN(x))），最后一层之后再接一个 LayerNorm 供策略头和价值头使用
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim,
                                                   nhead=embedding_dim // transformer_head_dim,
                                                   dim_feedforward=dim_feedforward, dropout=dropout,
                                                   batch_first=True, norm_first=True)
        self.transform_encoder = nn.TransformerEncoder(encoder_layer, num_layers, norm=nn.LayerNorm(embedding_dim))
        # 预测action的概率分布
        self.action_prob_dense = nn.Linear(embedding_dim, n_actions, bias=True)
        # 预测每个座位（相对当前行动玩家）的终局收益
        self.reward_value_dense = nn.Linear(embedding_dim, n_players, bias=True)

    def forward(self, x):
        x = self.env_embedding(x)
        x = self.transform_encoder(x)
        action_prob_logits = self.action_prob_dense(x[:, 0, :])
        reward_value = self.reward_value_dense(x[:, 1, :])
        return action_prob_logits, reward_value


class SplendorAlphaGoZero(nn.Module):
    def __init__(self,
                 n_observation=N_OBSERVATION,
                 n_actions=N_ACTIONS,
                 n_players=N_PLAYERS,
                 device='auto',
                 embedding_dim=256,
                 num_layers=6,
                 transformer_head_dim=64,
                 dropout=0.1,
                 embedding_init_rms='auto',
                 dim_feedforward=2048,
                 float32_matmul_precision=None,
                 batch_size=64,
                 base_learning_rate=1e-5,
                 max_learning_rate=1e-4,
                 step_size_up=1000,
                 step_size_down=5000,
                 l2_weight=0.,
                 value_loss_weight=1.,
                 transition_buffer_len=200000,
                 save_train_data=False,
                 default_train_file_path='data/splendor_history.txt',
                 num_inference_per_step=500,
                 num_data_print_per_inference=3,
                 **unused_kwargs):
        super().__init__()
        if unused_kwargs:
            logging.warning(f'SplendorAlphaGoZero ignored params: {sorted(unused_kwargs)}')
        assert n_observation == N_OBSERVATION and n_actions == N_ACTIONS and n_players == N_PLAYERS, \
            'n_observation / n_actions / n_players are fixed by splendor/rl_env.py'
        self.n_observation = n_observation
        self.n_actions = n_actions
        self.n_players = n_players
        self.device = torch.device(resolve_device(device))
        if float32_matmul_precision is not None:
            # 'high' lets float32 matmuls run on TF32 tensor cores (process-wide setting)
            torch.set_float32_matmul_precision(float32_matmul_precision)
        self.batch_size = batch_size
        self.value_loss_weight = value_loss_weight
        self.save_train_data = save_train_data
        self.default_train_file_path = default_train_file_path
        self.num_inference_per_step = num_inference_per_step
        self.num_data_print_per_inference = num_data_print_per_inference

        self.model = SplendorTransformerModel(n_actions, n_players, embedding_dim, num_layers, transformer_head_dim,
                                              dropout, embedding_init_rms, dim_feedforward).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=max_learning_rate, weight_decay=l2_weight)
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=base_learning_rate, max_lr=max_learning_rate,
                                                           step_size_up=step_size_up, step_size_down=step_size_down,
                                                           mode='triangular', cycle_momentum=False)
        self.reward_value_loss = torch.nn.MSELoss()

        self.memory_lock = Lock()
        self.memory = SimpleMemory(transition_buffer_len, 4)
        self.train_file = None
        if self.save_train_data:
            self.train_file = open(default_train_file_path, 'a', encoding='UTF-8')
        self.train_step_num = 0

    def close(self):
        if self.train_file is not None:
            self.train_file.flush()
            self.train_file.close()
            self.train_file = None

    def forward(self, x):
        return self.model(x)

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def predict_batch(self, observation_list, mask_list):
        """-> (action_probs[B, n_actions] masked+normalised, values[B, n_players]) as numpy."""
        self.model.eval()
        obs = torch.as_tensor(np.asarray(observation_list), dtype=torch.int64, device=self.device)
        mask = torch.as_tensor(np.asarray(mask_list), dtype=torch.bool, device=self.device)
        logits, values = self.model(obs)
        logits = logits.float().masked_fill(~mask, ILLEGAL_LOGIT)
        probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy(), values.float().cpu().numpy()

    def predict(self, observation, mask):
        probs, values = self.predict_batch([observation], [mask])
        return probs[0], values[0]

    # ------------------------------------------------------------------ training
    def store_transition(self, observation, action_probs, action_mask, reward_values, save_train_data=True):
        if self.train_file is not None and save_train_data:
            observation_str = ','.join(str(int(i)) for i in observation)
            action_prob_str = ','.join('%.5f' % prob for prob in action_probs)
            action_mask_str = ','.join('%d' % int(m) for m in action_mask)
            reward_value_str = ','.join('%.6f' % v for v in reward_values)
            self.train_file.write(f'{observation_str}\n{action_prob_str}\n{action_mask_str}\n{reward_value_str}\n\n')
        with self.memory_lock:
            self.memory.store([np.asarray(observation, dtype=np.int32), np.asarray(action_probs, dtype=np.float32),
                               np.asarray(action_mask, dtype=bool), np.asarray(reward_values, dtype=np.float32)])

    def flush_train_data(self):
        if self.train_file is not None:
            self.train_file.flush()

    def learn(self, observation_list=None, action_probs_list=None, action_mask_list=None, reward_value_list=None):
        if observation_list is None:
            with self.memory_lock:
                _, _, data_batch = self.memory.sample(self.batch_size)
            observation_list, action_probs_list, action_mask_list, reward_value_list = zip(*data_batch)

        observation_tensor = torch.as_tensor(np.asarray(observation_list), dtype=torch.int64, device=self.device)
        action_probs_tensor = torch.as_tensor(np.asarray(action_probs_list), dtype=torch.float32, device=self.device)
        action_mask_tensor = torch.as_tensor(np.asarray(action_mask_list), dtype=torch.bool, device=self.device)
        reward_value_tensor = torch.as_tensor(np.asarray(reward_value_list), dtype=torch.float32, device=self.device)

        self.model.train()
        action_prob_logits, reward_value_pred = self.model(observation_tensor)
        masked_logits = action_prob_logits.masked_fill(~action_mask_tensor, ILLEGAL_LOGIT)
        log_probs = torch.log_softmax(masked_logits, dim=1)
        action_probs_loss = -(action_probs_tensor * log_probs).sum(dim=1).mean()
        reward_value_loss = self.reward_value_loss(reward_value_pred, reward_value_tensor)
        over_all_loss = action_probs_loss + self.value_loss_weight * reward_value_loss
        if torch.isnan(over_all_loss):
            raise ValueError('loss is nan')

        self.optimizer.zero_grad()
        over_all_loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        self.train_step_num += 1
        if self.num_inference_per_step > 0 and self.train_step_num % self.num_inference_per_step == 0:
            predict_probs = torch.softmax(masked_logits, dim=1).detach().cpu().numpy()
            predict_values = reward_value_pred.detach().cpu().numpy()
            for i in range(min(self.num_data_print_per_inference, len(predict_probs))):
                target = np.asarray(action_probs_list[i])
                top = np.argsort(-target)[:5]
                logging.info('predictions:\n'
                             f'top_target_actions={top.tolist()}\n'
                             f'target_probs={",".join("%.3f" % target[a] for a in top)}\n'
                             f'predict_probs={",".join("%.3f" % predict_probs[i][a] for a in top)}\n'
                             f'reward_value={np.round(np.asarray(reward_value_list[i]), 3).tolist()}\n'
                             f'predict_reward_value={np.round(predict_values[i], 3).tolist()}')

        return action_probs_loss.item(), reward_value_loss.item()


def save_model(model, path):
    torch.save({'model': model.state_dict(), 'optimizer': model.optimizer.state_dict()}, path)
    logging.info(f'model saved to {path}')


def load_model(model, path, load_optimizer=True):
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
    if load_optimizer and 'optimizer' in checkpoint:
        try:
            model.optimizer.load_state_dict(checkpoint['optimizer'])
        except ValueError as e:
            logging.warning(f'optimizer state not loaded from {path}: {e}')
    logging.info(f'model loaded from {path}')


def splendor_train_data_generator(filename, epoch=1):
    """Reads the text dump written by SplendorAlphaGoZero.store_transition."""
    current_epoch = 1
    while True:
        with open(filename, 'r', encoding='UTF-8') as f:
            while True:
                lines = [f.readline() for _ in range(5)]
                if any(line == '' for line in lines[:4]):
                    break
                yield (np.asarray([int(v) for v in lines[0].rstrip().split(',')], dtype=np.int32),
                       np.asarray([float(v) for v in lines[1].rstrip().split(',')], dtype=np.float32),
                       np.asarray([int(v) for v in lines[2].rstrip().split(',')], dtype=bool),
                       np.asarray([float(v) for v in lines[3].rstrip().split(',')], dtype=np.float32))
        if 0 <= epoch <= current_epoch:
            break
        current_epoch += 1
