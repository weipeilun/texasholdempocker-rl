"""AlphaGoZero network for Splendor.

Same design as rl/models.TransformerAlphaGoZeroModel + rl/AlphaGoZero.AlphaGoZero (poker):
    * every observation field is an integer looked up in one shared embedding table
      (one vocab range per field group, fields of a group share it);
    * a factorised field-position embedding (attr + owner + slot) is ADDED to the token
      embedding (no concatenation, no sqrt(d) scaling), then a pre-norm TransformerEncoder with a
      final LayerNorm; the embedding is initialised at the scale of one sublayer's residual write;
    * learnable "starter" tokens in front of the sequence carry the heads:
      starter 0 -> action probability logits, starter 1 -> value;
    * SimpleMemory replay buffer, optional text dump of train data.

Differences:
    * value head: softmax over N_PLAYERS + 1 outcomes, "seat k (relative to the acting player) wins" and
      "nobody wins", trained with cross-entropy against the win shares (a draw: 0.25 each). The value of a
      seat is its predicted win share, a calibrated probability in [0, 1]. Every game ended by the rules has
      a winner or a draw, so the "nobody wins" class gets no target mass any more (kept so that checkpoints
      stay loadable); positions of truncated games carry value weight 0 and train the policy only;
    * the policy loss masks illegal actions before the softmax;
    * AdamW (decoupled weight decay, none on biases / LayerNorm), linear warmup then constant learning
      rate, gradient clipping;
    * every training sample is transformed by a random symmetry of the encoding (slot permutations,
      splendor.rl_env.SYMMETRY_GROUPS);
    * nn.TransformerEncoder deep-copies one layer num_layers times, so all layers would start with identical
      weights; every layer is re-initialised independently after construction.
"""

import logging
import math
import os
from threading import Lock

import numpy as np
import torch
from torch import nn

from tools.memories.simple_memory import SimpleMemory
from utils.torch_utils import PRE_NORM_FFN_WRITE_RMS, resolve_embedding_init_rms
from splendor.rl_env import (FIELD_GROUPS, VOCAB_SIZES, ATTRS, OWNERS, SLOTS, FIELD_ATTR_IDX, FIELD_OWNER_IDX,
                             FIELD_SLOT_IDX, N_ACTIONS, N_OBSERVATION, N_PLAYERS, SYMMETRY_GROUPS)

ILLEGAL_LOGIT = -1e4
# pinned staging buffer sets per model: up to PREDICTION_RING - 1 predictions in flight (predict_batch_async)
PREDICTION_RING = 3


def resolve_device(device):
    if device in (None, 'auto'):
        if torch.cuda.is_available():
            return 'cuda:0'
        if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'
    return device


def _entity_of_field():
    """field -> entity index (one per interchangeable slot of SYMMETRY_GROUPS) or -1 for a stand-alone field"""
    entity = [-1] * N_OBSERVATION
    for g, (obs_pos, _) in enumerate(SYMMETRY_GROUPS):
        for j in range(len(obs_pos)):
            for f in obs_pos[j]:
                entity[f] = (g, j)
    return entity


def build_token_layout():
    """Group the observation fields into tokens.

    Every slot of SYMMETRY_GROUPS (a market card, a reserved card, a noble) is ONE token: the scaled sum of its
    fields' value embeddings, whose position embedding is the same for every slot of the group (owner +
    entity type, no slot index). All other fields are one token each with position attr + owner + slot.
    Returns dict:
        field_token [N_OBSERVATION]  token of each field
        token_scale [T]              1 / sqrt(number of fields of the token)
        token_attr / token_owner / token_slot [T]  position embedding indices
        entity_tokens [g] -> [n_slots] token index of each slot of SYMMETRY_GROUPS[g]
    """
    entity = _entity_of_field()
    attrs = list(ATTRS) + ['card_entity', 'noble_entity']
    tokens, field_token, entity_tokens = dict(), [], [[None] * len(obs_pos) for obs_pos, _ in SYMMETRY_GROUPS]
    token_attr, token_owner, token_slot, token_fields = [], [], [], []
    for f, (attr, owner, slot) in enumerate(zip(FIELD_GROUPS, (OWNERS[i] for i in FIELD_OWNER_IDX), (SLOTS[i] for i in FIELD_SLOT_IDX))):
        key = ('entity',) + entity[f] if entity[f] != -1 else ('field', f)
        if key not in tokens:
            tokens[key] = len(tokens)
            if entity[f] != -1:
                g, j = entity[f]
                entity_tokens[g][j] = tokens[key]
                token_attr.append(attrs.index('noble_entity' if attr.startswith('noble_req') else 'card_entity'))
                token_slot.append(SLOTS.index('none'))
            else:
                token_attr.append(attrs.index(attr))
                token_slot.append(SLOTS.index(slot))
            token_owner.append(OWNERS.index(owner))
            token_fields.append(0)
        field_token.append(tokens[key])
        token_fields[tokens[key]] += 1
    return dict(field_token=field_token, token_scale=[1. / math.sqrt(n) for n in token_fields],
                token_attr=token_attr, token_owner=token_owner, token_slot=token_slot,
                entity_tokens=entity_tokens, num_attrs=len(attrs))


class SplendorEnvEmbedding(nn.Module):
    """token embedding + factorised position embedding, summed.

    * field_embedding: one shared table; each field's integer value is offset by its vocab group's start,
      so fields of the same group share rows and different groups never collide. Rows [0, num_starters)
      are the starter tokens.
    * tokens (build_token_layout): an entity (card slot, reserved slot, noble slot) is one token, the sum of
      its fields' embeddings / sqrt(#fields); every other field is a token of its own. 290 fields -> 102 tokens.
    * position = E_attr + E_owner + E_slot of the token. Interchangeable slots share their position (no
      slot index), so the network treats the slots of a group as a set: it is exactly invariant to
      permuting them (with the pointer action heads of SplendorTransformerModel, exactly equivariant).
    * no sqrt(d) scaling. The summed embedding is the first write into the pre-norm residual stream; it is
      initialised with total RMS `init_rms` (default: the RMS one FFN sublayer writes at init,
      utils.torch_utils.PRE_NORM_FFN_WRITE_RMS), split evenly between content and position.
    """

    def __init__(self, embedding_dim, num_starters, init_rms=PRE_NORM_FFN_WRITE_RMS):
        super().__init__()
        self.num_starters = num_starters
        layout = build_token_layout()
        self.layout = layout
        self.num_tokens = len(layout['token_attr'])

        group_offsets = dict()
        current = num_starters
        for group in ATTRS:
            group_offsets[group] = current
            current += VOCAB_SIZES[group]
        self.register_buffer('field_start_idx_array', torch.tensor([group_offsets[g] for g in FIELD_GROUPS], dtype=torch.int64))
        self.register_buffer('starter_idx_array', torch.arange(0, num_starters, dtype=torch.int64))
        self.field_embedding = nn.Embedding(num_embeddings=current, embedding_dim=embedding_dim)
        # [T, F] fields -> tokens: token = sum of its fields / sqrt(#fields) (a fixed matrix, deterministic)
        pool = torch.zeros(self.num_tokens, N_OBSERVATION)
        for f, t in enumerate(layout['field_token']):
            pool[t, f] = layout['token_scale'][t]
        self.register_buffer('field_to_token', pool)

        self.register_buffer('token_attr_idx', torch.tensor(layout['token_attr'], dtype=torch.int64))
        self.register_buffer('token_owner_idx', torch.tensor(layout['token_owner'], dtype=torch.int64))
        self.register_buffer('token_slot_idx', torch.tensor(layout['token_slot'], dtype=torch.int64))
        self.attr_embedding = nn.Embedding(num_embeddings=layout['num_attrs'], embedding_dim=embedding_dim)
        self.owner_embedding = nn.Embedding(num_embeddings=len(OWNERS), embedding_dim=embedding_dim)
        self.slot_embedding = nn.Embedding(num_embeddings=len(SLOTS), embedding_dim=embedding_dim)

        if init_rms is not None:
            nn.init.normal_(self.field_embedding.weight, mean=0., std=init_rms / math.sqrt(2.))
            for embedding in (self.attr_embedding, self.owner_embedding, self.slot_embedding):
                nn.init.normal_(embedding.weight, mean=0., std=init_rms / math.sqrt(6.))

    def position_embedding(self):
        """[num_starters + num_tokens, embedding_dim]; zeros for the starter tokens."""
        token_position = (self.attr_embedding(self.token_attr_idx)
                          + self.owner_embedding(self.token_owner_idx)
                          + self.slot_embedding(self.token_slot_idx))
        starter_position = token_position.new_zeros((self.num_starters, token_position.shape[1]))
        return torch.cat((starter_position, token_position), dim=0)

    def forward(self, x):
        batch_size = x.shape[0]
        fields = self.field_embedding(x.long() + self.field_start_idx_array)            # [B, F, d]
        tokens = torch.matmul(self.field_to_token.to(fields.dtype), fields)             # [B, T, d]
        starters = self.field_embedding(self.starter_idx_array).unsqueeze(0).expand(batch_size, -1, -1)
        # [B, L, d] + [L, d] broadcast
        return torch.cat((starters, tokens), dim=1) + self.position_embedding()


class SplendorTransformerModel(nn.Module):
    """Pre-norm transformer over [policy starter, value starter, tokens].

    Action heads: a slot action is scored from its slot's token (pointer head): reserve / buy of a market
    card from that card's token, buy of reserved card i from the acting player's reserved token i, noble
    slot j from noble token j. The other actions (take tokens, return a token) come from starter 0.
    The outcome head (N_PLAYERS + 1 logits) reads starter 1.
    """

    def __init__(self, n_actions=N_ACTIONS, n_players=N_PLAYERS, embedding_dim=256,
                 num_layers=6, transformer_head_dim=64, dropout=0.1, embedding_init_rms='auto', dim_feedforward=2048):
        super().__init__()
        assert embedding_dim % transformer_head_dim == 0, \
            f'embedding_dim({embedding_dim}) must be divisible by transformer_head_dim({transformer_head_dim}).'
        self.embedding_dim = embedding_dim
        self.num_starters = 2

        self.env_embedding = SplendorEnvEmbedding(embedding_dim, num_starters=self.num_starters,
                                                  init_rms=resolve_embedding_init_rms(embedding_init_rms))
        # pre-norm（x = x + f(LN(x))），最后一层之后再接一个 LayerNorm 供策略头和价值头使用
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim,
                                                   nhead=embedding_dim // transformer_head_dim,
                                                   dim_feedforward=dim_feedforward, dropout=dropout,
                                                   batch_first=True, norm_first=True)
        self.transform_encoder = nn.TransformerEncoder(encoder_layer, num_layers, norm=nn.LayerNorm(embedding_dim),
                                                       enable_nested_tensor=False)
        # TransformerEncoder deep-copies encoder_layer: without this every layer starts with the same weights
        for layer in self.transform_encoder.layers:
            reinit_encoder_layer(layer)

        # pointer heads: groups of SYMMETRY_GROUPS with slot actions, one head per kind of slot
        layout = self.env_embedding.layout
        pointer_tokens, pointer_actions, head_of_group, self.pointer_kinds = [], [], [], []
        kinds = dict()
        for g, (obs_pos, act_pos) in enumerate(SYMMETRY_GROUPS):
            if act_pos is None:
                continue
            kind = (FIELD_GROUPS[obs_pos[0][0]], act_pos.shape[1])   # (first field's attr, actions per slot)
            if kind not in kinds:
                kinds[kind] = len(kinds)
                self.pointer_kinds.append(act_pos.shape[1])
            for j in range(len(obs_pos)):
                pointer_tokens.append(self.num_starters + layout['entity_tokens'][g][j])
                pointer_actions.append(list(act_pos[j]))
                head_of_group.append(kinds[kind])
        self.pointer_heads = nn.ModuleList([nn.Linear(embedding_dim, width, bias=True) for width in self.pointer_kinds])
        slot_actions = sorted(a for actions in pointer_actions for a in actions)
        global_actions = [a for a in range(n_actions) if a not in set(slot_actions)]
        self.action_prob_dense = nn.Linear(embedding_dim, len(global_actions), bias=True)
        # order in which forward() concatenates logits: global actions, then each kind's slots in order
        produced = list(global_actions)
        self.pointer_groups = list()   # per kind: (token indices tensor name, number of slots)
        for k in range(len(self.pointer_kinds)):
            idx = [i for i, h in enumerate(head_of_group) if h == k]
            self.register_buffer(f'pointer_tokens_{k}', torch.tensor([pointer_tokens[i] for i in idx], dtype=torch.int64))
            produced += [a for i in idx for a in pointer_actions[i]]
        assert sorted(produced) == list(range(n_actions))
        self.register_buffer('action_order', torch.tensor([produced.index(a) for a in range(n_actions)], dtype=torch.int64))
        # 终局结果的 logits：座位 k（相对当前行动玩家）获胜 x n_players，以及无人获胜（blocked / truncated）
        self.reward_value_dense = nn.Linear(embedding_dim, n_players + 1, bias=True)

    def forward(self, x):
        """-> (action logits [B, n_actions], outcome logits [B, n_players + 1])"""
        x = self.env_embedding(x)
        x = self.transform_encoder(x)
        parts = [self.action_prob_dense(x[:, 0, :])]
        for k, head in enumerate(self.pointer_heads):
            tokens = x[:, getattr(self, f'pointer_tokens_{k}'), :]          # [B, n_slots, d]
            parts.append(head(tokens).reshape(x.shape[0], -1))             # [B, n_slots * width]
        action_prob_logits = torch.cat(parts, dim=1)[:, self.action_order]
        outcome_logits = self.reward_value_dense(x[:, 1, :])
        return action_prob_logits, outcome_logits


def reinit_encoder_layer(layer):
    """fresh PyTorch default initialisation of one nn.TransformerEncoderLayer (attention: xavier uniform
    in-projection, zero biases; linears: kaiming uniform; LayerNorms: ones / zeros)"""
    layer.self_attn._reset_parameters()
    layer.self_attn.out_proj.reset_parameters()
    nn.init.zeros_(layer.self_attn.out_proj.bias)   # as MultiheadAttention._reset_parameters does
    for module in (layer.linear1, layer.linear2, layer.norm1, layer.norm2):
        module.reset_parameters()


def outcome_targets(reward_values):
    """win shares relative to the acting player [B, n_players] -> outcome distribution [B, n_players + 1]
    (last entry: nobody wins, the remaining probability mass)"""
    no_winner = (1. - reward_values.sum(dim=1, keepdim=True)).clamp_min(0.)
    return torch.cat((reward_values, no_winner), dim=1)


class SymmetryAugmenter(nn.Module):
    """Random slot permutations (splendor.rl_env.SYMMETRY_GROUPS) of a batch, on the batch's device."""

    def __init__(self):
        super().__init__()
        self.groups = list()
        for k, (obs_pos, act_pos) in enumerate(SYMMETRY_GROUPS):
            self.register_buffer(f'obs_pos_{k}', torch.as_tensor(obs_pos, dtype=torch.int64), persistent=False)
            if act_pos is not None:
                self.register_buffer(f'act_pos_{k}', torch.as_tensor(act_pos, dtype=torch.int64), persistent=False)
            self.groups.append((f'obs_pos_{k}', None if act_pos is None else f'act_pos_{k}'))

    def indices(self, batch_size, device, perms=None):
        """-> (obs_index [B, N_OBSERVATION], action_index [B, N_ACTIONS]) of one random symmetry per row;
        perms (tests): one [B, n_slots] permutation tensor per group instead of random ones"""
        obs_index = torch.arange(N_OBSERVATION, device=device).expand(batch_size, -1).clone()
        action_index = torch.arange(N_ACTIONS, device=device).expand(batch_size, -1).clone()
        for k, (obs_name, act_name) in enumerate(self.groups):
            obs_pos = getattr(self, obs_name)
            perm = perms[k] if perms is not None else torch.argsort(torch.rand(batch_size, obs_pos.shape[0], device=device), dim=1)
            obs_index[:, obs_pos.reshape(-1)] = obs_pos[perm].reshape(batch_size, -1)
            if act_name is not None:
                act_pos = getattr(self, act_name)
                action_index[:, act_pos.reshape(-1)] = act_pos[perm].reshape(batch_size, -1)
        return obs_index, action_index

    def forward(self, observation, action_probs, action_mask):
        obs_index, action_index = self.indices(observation.shape[0], observation.device)
        return observation.gather(1, obs_index), action_probs.gather(1, action_index), action_mask.gather(1, action_index)


class PendingPrediction:
    """Handle of a prediction started by SplendorAlphaGoZero.predict_batch_async."""

    def __init__(self, done_event, host_out, n_actions):
        self.done_event, self.host_out, self.n_actions = done_event, host_out, n_actions

    def ready(self):
        return self.done_event is None or self.done_event.query()

    def result(self):
        """-> (action_probs, values) views of a pinned host buffer, reused PREDICTION_RING predictions later"""
        if self.done_event is not None:
            self.done_event.synchronize()
        return self.host_out[:, :self.n_actions], self.host_out[:, self.n_actions:]


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
                 inference_bf16_min_batch=None,
                 inference_cuda_graph_batch_sizes=None,
                 batch_size=64,
                 learning_rate=2e-4,
                 lr_warmup_steps=200,
                 weight_decay=0.1,
                 adam_betas=(0.9, 0.99),
                 grad_clip_norm=1.,
                 value_loss_weight=0.25,
                 train_symmetry_augmentation=True,
                 transition_buffer_len=200000,
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
        # inference (predict_batch) runs in bfloat16 autocast for batches of at least this size; None = never.
        # bf16 is faster only for larger batches (small batches are bound by kernel launches, not FLOPs)
        self.inference_bf16_min_batch = inference_bf16_min_batch
        # batch sizes captured with CUDA graphs by the predict process (enable_cuda_graphs); None = eager only
        self.inference_cuda_graph_batch_sizes = inference_cuda_graph_batch_sizes
        if float32_matmul_precision is not None:
            # 'high' lets float32 matmuls run on TF32 tensor cores (process-wide setting)
            torch.set_float32_matmul_precision(float32_matmul_precision)
        self.batch_size = batch_size
        self.value_loss_weight = value_loss_weight
        self.num_inference_per_step = num_inference_per_step
        self.num_data_print_per_inference = num_data_print_per_inference

        self.model = SplendorTransformerModel(n_actions, n_players, embedding_dim, num_layers, transformer_head_dim,
                                              dropout, embedding_init_rms, dim_feedforward).to(self.device)
        # decoupled weight decay on matrices and embeddings, none on biases and LayerNorm parameters
        decay = [p for p in self.model.parameters() if p.dim() >= 2]
        no_decay = [p for p in self.model.parameters() if p.dim() < 2]
        self.optimizer = torch.optim.AdamW([{'params': decay, 'weight_decay': weight_decay},
                                            {'params': no_decay, 'weight_decay': 0.}],
                                           lr=learning_rate, betas=tuple(adam_betas))
        warmup = max(1, int(lr_warmup_steps))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda step: min(1., (step + 1) / warmup))
        self.grad_clip_norm = grad_clip_norm
        self.augmenter = SymmetryAugmenter().to(self.device) if train_symmetry_augmentation else None

        self.memory_lock = Lock()
        # replay buffer of (observation, policy target, legal mask, value target, value weight); the train data
        # file is written by the training loop (rl/splendor_data.TrainDataWriter)
        self.memory = SimpleMemory(transition_buffer_len, 5)
        self.train_step_num = 0

    def close(self):
        pass

    def forward(self, x):
        return self.model(x)

    # ------------------------------------------------------------------ inference
    def _use_bf16(self, batch_size):
        return (self.inference_bf16_min_batch is not None and self.device.type == 'cuda'
                and batch_size >= self.inference_bf16_min_batch)

    def _infer(self, obs, mask, use_bf16):
        """obs int64 [B, N_OBSERVATION], mask bool [B, n_actions] on the device -> float32 [B, n_actions + n_players]"""
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
            logits, outcome_logits = self.model(obs)
        logits = logits.float().masked_fill(~mask, ILLEGAL_LOGIT)
        probs = torch.softmax(logits, dim=1)
        values = torch.softmax(outcome_logits.float(), dim=1)[:, :self.n_players]   # predicted win shares
        return torch.cat((probs, values), dim=1)

    @torch.no_grad()
    def enable_cuda_graphs(self, batch_sizes=(8, 16, 32, 64, 128, 256)):
        """Capture the inference forward once per batch size with CUDA Graphs.

        An eager forward launches ~150 small kernels from the CPU, which dominates small-batch latency and
        slows down further when the CPU is busy; a graph replay is a single launch. predict_batch pads a
        batch to the next captured size. The captured graphs read the model's weight tensors in place, so
        weights changed later with load_state_dict / training steps are picked up by the replays.
        """
        if self.device.type != 'cuda':
            return
        self.model.eval()
        self._graphs = dict()
        # one memory pool shared by all graphs (replays never overlap); capture the largest first so the
        # smaller graphs fit into memory the pool already holds
        pool = torch.cuda.graph_pool_handle()
        for batch_size in sorted(set(int(b) for b in batch_sizes), reverse=True):
            static_obs = torch.zeros((batch_size, self.n_observation), dtype=torch.long, device=self.device)
            static_mask = torch.ones((batch_size, self.n_actions), dtype=torch.bool, device=self.device)
            use_bf16 = self._use_bf16(batch_size)
            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                for _ in range(3):   # warm up allocations and lazy initialisation outside the capture
                    self._infer(static_obs, static_mask, use_bf16)
            torch.cuda.current_stream(self.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                static_out = self._infer(static_obs, static_mask, use_bf16)
            self._graphs[batch_size] = (graph, static_obs, static_mask, static_out)
        torch.cuda.synchronize(self.device)
        logging.info(f'CUDA graphs captured for inference batch sizes {sorted(self._graphs)}')

    def _staging(self, batch_size):
        """-> the next (obs uint8, mask bool, out float32) set of pinned host buffers of a small ring.

        Copies from pageable memory may synchronise with the stream, so inputs are staged in pinned memory
        and every copy is truly asynchronous. Each prediction uses the next set of the ring, so up to
        PREDICTION_RING - 1 predictions can be in flight while the results of an earlier one are read.
        """
        ring = getattr(self, '_staging_ring', None)
        if ring is None or ring[0][0].shape[0] < batch_size:
            cap = max([batch_size, 256] + list(getattr(self, '_graphs', None) or ()))
            ring = self._staging_ring = [(torch.empty((cap, self.n_observation), dtype=torch.uint8, pin_memory=True),
                                          torch.empty((cap, self.n_actions), dtype=torch.bool, pin_memory=True),
                                          torch.empty((cap, self.n_actions + self.n_players), dtype=torch.float32, pin_memory=True))
                                         for _ in range(PREDICTION_RING)]
            self._staging_next = 0
        k = self._staging_next
        self._staging_next = (k + 1) % len(ring)
        return ring[k]

    @torch.no_grad()
    def predict_batch_async(self, observation_list, mask_list):
        """Start a batched prediction and return at once with a PendingPrediction; .result() waits for it.

        On CUDA the inputs are copied, the forward (a CUDA graph replay when one is captured for a batch size
        >= B, padded) and the device->host copy into a pinned buffer are only queued, so the caller can do CPU
        work while the GPU computes. Up to PREDICTION_RING - 1 predictions may be pending at a time (they run
        in order on one stream); a result's arrays are reused PREDICTION_RING predictions later.
        """
        self.model.eval()
        obs_np = np.asarray(observation_list)
        mask_np = np.asarray(mask_list, dtype=bool)
        batch_size = len(obs_np)
        staging = self._staging(batch_size) if self.device.type == 'cuda' else None
        # small integer types on the host -> device copy, widened on the GPU; one device->host copy for both outputs
        if staging is not None and obs_np.dtype == np.uint8:
            obs_host, mask_host = staging[0][:batch_size], staging[1][:batch_size]
            obs_host.numpy()[:] = obs_np
            mask_host.numpy()[:] = mask_np
            obs = obs_host.to(self.device, non_blocking=True)
            mask = mask_host.to(self.device, non_blocking=True)
        else:
            obs = torch.from_numpy(np.ascontiguousarray(obs_np)).to(self.device, non_blocking=True)
            mask = torch.from_numpy(np.ascontiguousarray(mask_np)).to(self.device, non_blocking=True)
        graphs = getattr(self, '_graphs', None)
        graph_size = min((b for b in graphs if b >= batch_size), default=None) if graphs else None
        if graph_size is not None:
            graph, static_obs, static_mask, static_out = graphs[graph_size]
            static_obs[:batch_size].copy_(obs)       # rows past batch_size keep stale inputs; their outputs are unused
            static_mask[:batch_size].copy_(mask)
            graph.replay()
            out = static_out[:batch_size]
        else:
            out = self._infer(obs.long(), mask, self._use_bf16(batch_size))
        if self.device.type != 'cuda':
            return PendingPrediction(None, out.numpy(), self.n_actions)
        host = staging[2][:batch_size]
        host.copy_(out, non_blocking=True)
        done = torch.cuda.Event()
        done.record()
        return PendingPrediction(done, host.numpy(), self.n_actions)

    def predict_batch(self, observation_list, mask_list):
        """-> (action_probs[B, n_actions] masked+normalised, values[B, n_players]) as numpy."""
        probs, values = self.predict_batch_async(observation_list, mask_list).result()
        return probs.copy(), values.copy()   # the pinned buffer is reused by the next call

    def predict(self, observation, mask):
        probs, values = self.predict_batch([observation], [mask])
        return probs[0], values[0]

    # ------------------------------------------------------------------ training
    def store_transition(self, observation, action_probs, action_mask, reward_values, value_weight=1.):
        """add one sample to the replay buffer (value_weight 0: a position of a truncated game, no value target)"""
        with self.memory_lock:
            self.memory.store([np.asarray(observation, dtype=np.int32), np.asarray(action_probs, dtype=np.float32),
                               np.asarray(action_mask, dtype=bool), np.asarray(reward_values, dtype=np.float32),
                               np.float32(value_weight)])

    def learn(self, observation_list=None, action_probs_list=None, action_mask_list=None, reward_value_list=None,
              value_weight_list=None):
        if observation_list is None:
            with self.memory_lock:
                _, _, data_batch = self.memory.sample(self.batch_size)
            observation_list, action_probs_list, action_mask_list, reward_value_list, value_weight_list = zip(*data_batch)
        if value_weight_list is None:
            value_weight_list = np.ones(len(observation_list), dtype=np.float32)

        observation_tensor = torch.as_tensor(np.asarray(observation_list), dtype=torch.int64, device=self.device)
        action_probs_tensor = torch.as_tensor(np.asarray(action_probs_list), dtype=torch.float32, device=self.device)
        action_mask_tensor = torch.as_tensor(np.asarray(action_mask_list), dtype=torch.bool, device=self.device)
        reward_value_tensor = torch.as_tensor(np.asarray(reward_value_list), dtype=torch.float32, device=self.device)
        value_weight_tensor = torch.as_tensor(np.asarray(value_weight_list), dtype=torch.float32, device=self.device)

        if self.augmenter is not None:
            observation_tensor, action_probs_tensor, action_mask_tensor = self.augmenter(
                observation_tensor, action_probs_tensor, action_mask_tensor)

        self.model.train()
        action_prob_logits, outcome_logits = self.model(observation_tensor)
        masked_logits = action_prob_logits.masked_fill(~action_mask_tensor, ILLEGAL_LOGIT)
        log_probs = torch.log_softmax(masked_logits, dim=1)
        action_probs_loss = -(action_probs_tensor * log_probs).sum(dim=1).mean()
        # value cross-entropy over the samples with a result (truncated games: weight 0)
        value_ce = -(outcome_targets(reward_value_tensor) * torch.log_softmax(outcome_logits, dim=1)).sum(dim=1)
        reward_value_loss = (value_ce * value_weight_tensor).sum() / value_weight_tensor.sum().clamp_min(1.)
        over_all_loss = action_probs_loss + self.value_loss_weight * reward_value_loss
        if torch.isnan(over_all_loss):
            raise ValueError('loss is nan')

        self.optimizer.zero_grad(set_to_none=True)
        over_all_loss.backward()
        if self.grad_clip_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        self.optimizer.step()
        self.scheduler.step()
        reward_value_pred = torch.softmax(outcome_logits.detach(), dim=1)[:, :self.n_players]

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

    @torch.no_grad()
    def evaluate_samples(self, observations, action_probs, action_masks, reward_values, value_weights=None, batch_size=4096):
        """Loss metrics without training (eval mode, no augmentation).

        value_ce: outcome cross-entropy (the training loss); value_mse: mean squared error of the predicted win
        shares; value_mse_constant: MSE of always predicting the mean win shares of these samples (a model
        that has learned nothing about the position); winner_acc: the seat with the highest predicted win
        share is a winner, over games with a single-winner-or-shared result; policy_ce / target_entropy:
        policy loss and its floor. Value metrics use only samples with value weight 1 (games with a result)."""
        self.model.eval()
        weights_all = np.ones(len(observations), dtype=np.float32) if value_weights is None else np.asarray(value_weights, dtype=np.float32)
        with_value = weights_all > 0
        n = len(observations)
        n_value = int(with_value.sum())
        sums = dict(value_ce=0., value_mse=0., policy_ce=0., target_entropy=0.)
        correct, decided = 0, 0
        values_all = np.asarray(reward_values, dtype=np.float32)
        for i in range(0, n, batch_size):
            obs = torch.as_tensor(np.asarray(observations[i:i + batch_size]), dtype=torch.int64, device=self.device)
            target = torch.as_tensor(np.asarray(action_probs[i:i + batch_size]), dtype=torch.float32, device=self.device)
            mask = torch.as_tensor(np.asarray(action_masks[i:i + batch_size]), dtype=torch.bool, device=self.device)
            values = torch.as_tensor(values_all[i:i + batch_size], device=self.device)
            weight = torch.as_tensor(weights_all[i:i + batch_size], device=self.device)
            logits, outcome_logits = self.model(obs)
            log_probs = torch.log_softmax(logits.float().masked_fill(~mask, ILLEGAL_LOGIT), dim=1)
            sums['policy_ce'] += -(target * log_probs).sum().item()
            sums['target_entropy'] += -(target * torch.log(target.clamp_min(1e-12))).sum().item()
            outcome_log_probs = torch.log_softmax(outcome_logits.float(), dim=1)
            sums['value_ce'] += -((outcome_targets(values) * outcome_log_probs).sum(dim=1) * weight).sum().item()
            pred = outcome_log_probs.exp()[:, :self.n_players]
            sums['value_mse'] += (((pred - values) ** 2).mean(dim=1) * weight).sum().item()
            has_winner = (values.max(dim=1).values > 0.26) & (weight > 0)   # a draw (0.25 each) has no winner to hit
            winner_hit = values.gather(1, pred.argmax(dim=1, keepdim=True)).squeeze(1) > 0
            correct += (winner_hit & has_winner).sum().item()
            decided += has_winner.sum().item()
        metrics = {k: sums[k] / max(n, 1) for k in ('policy_ce', 'target_entropy')}
        metrics.update({k: sums[k] / max(n_value, 1) for k in ('value_ce', 'value_mse')})
        valued = values_all[with_value]
        metrics['value_mse_constant'] = float(((valued - valued.mean(axis=0)) ** 2).mean()) if n_value else 0.
        metrics['winner_acc'] = correct / max(decided, 1)
        metrics['num_samples'] = n
        metrics['num_value_samples'] = n_value
        return metrics

    def sample_memory(self, n):
        """n random samples of the replay buffer as (observations, action_probs, masks, values, value weights) arrays"""
        with self.memory_lock:
            _, _, data_batch = self.memory.sample(n)
        return tuple(np.asarray(x) for x in zip(*data_batch))

    def memory_size(self):
        return self.memory.capacity if self.memory.memory_full else self.memory.current_data_idx


# ---------------------------------------------------------------------------- checkpoints
# Parameters that decide the network's tensor shapes: a full resume requires them to be identical.
ARCHITECTURE_KEYS = ('embedding_dim', 'num_layers', 'transformer_head_dim', 'dim_feedforward')


def architecture_of(model_param_dict):
    defaults = {'embedding_dim': 256, 'num_layers': 6, 'transformer_head_dim': 64, 'dim_feedforward': 2048}
    return {k: model_param_dict.get(k, defaults[k]) for k in ARCHITECTURE_KEYS}


def save_checkpoint(path, payload):
    """torch.save to path + '.tmp', then rename: a checkpoint file is never half written"""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp = path + '.tmp'
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def model_state_of(checkpoint):
    """the network weights of a checkpoint (format 2, or the {'model', 'optimizer'} dicts of earlier versions)"""
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        return checkpoint['model']
    return checkpoint


def init_weights_from(model, path, strict=True):
    """Initialise only the network weights from a checkpoint (no optimizer, scheduler, counters or data).

    strict: every tensor of the model must be in the checkpoint with the same shape and the checkpoint must
    hold nothing else, otherwise ValueError. strict=False loads the matching tensors and keeps the fresh
    initialisation of the others (all differences are logged)."""
    from rl.splendor_data import versions_mismatch   # (no import cycle at module load)
    from splendor.rl_env import version_info
    checkpoint = load_checkpoint(path)
    state = model_state_of(checkpoint)
    own = model.state_dict()
    missing = sorted(k for k in own if k not in state)
    unexpected = sorted(k for k in state if k not in own)
    mismatched = sorted(k for k in own if k in state and tuple(state[k].shape) != tuple(own[k].shape))
    problems = []
    if missing:
        problems.append(f'{len(missing)} missing (e.g. {missing[:3]})')
    if unexpected:
        problems.append(f'{len(unexpected)} unexpected (e.g. {unexpected[:3]})')
    if mismatched:
        problems.append(f'{len(mismatched)} with other shapes (e.g. {mismatched[:3]})')
    if problems and strict:
        raise ValueError(f'cannot initialise the weights from {path}: tensors {"; ".join(problems)} '
                         f'(set init_weights_strict: false to load only the matching tensors)')
    versions = checkpoint.get('versions') if isinstance(checkpoint, dict) else None
    if versions is None:
        logging.warning(f'{path} records no rules / observation / action versions (saved by an older pipeline); '
                        f'its weights are used as initialisation only')
    else:
        mismatch = versions_mismatch({k: v for k, v in versions.items() if k != 'discard_over_limit'},
                                     {k: v for k, v in version_info().items() if k != 'discard_over_limit'})
        if mismatch:
            logging.warning(f'{path} was trained for other versions ({"; ".join(mismatch)}); its weights are used as initialisation only')
    loadable = {k: v for k, v in state.items() if k in own and k not in mismatched}
    model.load_state_dict(loadable, strict=False)
    if problems:
        logging.warning(f'weights initialised from {path} except: {"; ".join(problems)}')
    else:
        logging.info(f'weights initialised from {path} (all {len(own)} tensors)')
    return {'missing': missing, 'unexpected': unexpected, 'mismatched': mismatched}
