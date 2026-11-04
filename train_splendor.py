"""Splendor training entry, modelled on test_train.py.

Same skeleton as test_train.py:
    main: load default params + a search grid, pick `num_choices` distinct parameter
          combinations, run every combination in its own train_process (spawned, joined);
    train_process: train the AlphaGoZero model, evaluate the new model against the best
          one with MCTS games played by multiple processes sharing batched model
          predictions, decide whether the new model replaces the best one.

Same training method as the poker pipeline (main.py / env/workflow.py):
    MCTS self-play (rl/splendor_mcts.py, ported from rl/MCTS.py) produces
    (observation, MCTS visit distribution, legal mask, final reward) samples, the
    transformer policy/value network (rl/splendor_model.py) learns from a replay buffer
    (+ optional historical data file), and the new model is promoted only if it beats the
    best one in evaluation games. Predictions are batched in a predict process; each game
    worker process drives many games in one thread (searches are generators, requests travel
    in blocks through shared memory).

Splendor has no external train data, so every iteration of train_process is
    self-play -> train -> (every N iterations) eval new vs best.

Persistent game pool: the game worker processes and the predict process are started once per experiment and
host three model slots (current = latest weights for self-play, best and new for evaluation). Each iteration
only publishes new weights. Self-play never pauses: the next games start while the slowest games of the
previous iteration are still running, and an iteration is simply the next num_self_play_games_per_iteration
finished games. Evaluation games run in the same pool, alongside self-play.

Experiments: every run writes into its own directory (experiments/<experiment_name>/): config.yml, the
training data file, checkpoints/latest.pth (full training state), checkpoints/best.pth, train.log.
    new experiment:   python train_splendor.py --mode single [--default-config ...]
                      (init_weights_path: initialise only the network weights from any checkpoint)
    exact resume:     python train_splendor.py --resume experiments/<experiment_name>
                      (strict: versions, architecture and every tensor key must match)

Usage (from the repository root):
    python train_splendor.py                  # parameter search, config/train_splendor.yml
    python train_splendor.py --mode single    # one run with config/splendor_default.yml only
    python train_splendor.py --mode single --smoke   # tiny CPU run to check the pipeline
"""

import argparse
import collections
import copy
import logging
import math
import os
import random
import selectors
import struct
import threading
import time
from queue import Empty

import numpy as np
import torch
import torch.multiprocessing as mp
import yaml

from tools import interrupt
from tools.param_parser import update_concurrent, choose_params_concurrent
from utils.torch_utils import get_state_dict_from_model
from rl.splendor_data import (CHECKPOINT_FORMAT_VERSION, SEED_DEAL, SEED_EVAL_GAME, SEED_PLAYOUT_CAP, SEED_SAMPLES,
                              SEED_SEARCH, SEED_SYMMETRY, SEED_TRAIN_GAME, SEED_TRAINING, TrainDataWriter, derive_seed,
                              read_samples, truncate_data_file, versions_mismatch)
from rl.splendor_mcts import SplendorMCTS
from rl.splendor_model import (SplendorAlphaGoZero, architecture_of, init_weights_from, load_checkpoint, resolve_device,
                               save_checkpoint, PREDICTION_RING)
from splendor.rl_env import (SplendorRLEnv, to_relative, N_ACTIONS, N_OBSERVATION, N_PLAYERS, VOCAB_SIZES,
                             random_symmetry_indices, version_info)

LOG_FORMAT = '%(asctime)s.%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d,%H:%M:%S'

# 评估时新/旧模型各占两个座位。相邻两局使用同一种子、互补的座位安排，抵消座位与发牌的影响。
EVAL_SEAT_ARRANGEMENTS = [(1, 0, 1, 0), (0, 1, 0, 1), (1, 1, 0, 0), (0, 0, 1, 1), (1, 0, 0, 1), (0, 1, 1, 0)]
BEST_MODEL_IDX, NEW_MODEL_IDX = 0, 1
# model slots of the persistent predict process
SLOT_CURRENT, SLOT_BEST, SLOT_NEW = 0, 1, 2
NUM_SLOTS = 3
# parameters of earlier pipelines that no longer exist (an experiment directory replaces them)
REMOVED_PARAMS = ('train_data_path', 'train_with_historical_data', 'historical_data_mode', 'model_init_checkpoint_path',
                  'model_save_checkpoint_path_format', 'model_best_checkpoint_path_format',
                  'model_eval_snapshot_path_format', 'skip_unfinished_games')


def init_logging(log_level, log_file=None):
    """stderr, plus log_file (the experiment's train.log; every process appends to it) when given"""
    logging.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    root = logging.getLogger()
    root.setLevel(log_level)
    if log_file and not any(isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(log_file)
                            for h in root.handlers):
        handler = logging.FileHandler(log_file, encoding='UTF-8')
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
        root.addHandler(handler)


# ============================================================================ game
def play_game_iter(game_id, seed, seat_models, params, mode, should_stop=None):
    """Play one 4-player game with MCTS at every decision, as a generator: yields (model_id, request bytes)
    for every network evaluation (model_id = seat_models[acting seat]) and expects the reply bytes to be
    sent back (wire format, see "wire format" below); returns the result dict, or None when interrupted.

    All randomness comes from the game seed (rl/splendor_data.derive_seed): the deal and card draws, the
    full / fast search choice, and per decision index the search's Gumbel noise and sampled draws. A game
    plays the same whichever worker runs it (up to floating-point differences of batched evaluation).

    Gumbel AlphaZero (rl/splendor_mcts.py): the played move is the action selected by Sequential
    Halving with Gumbel, the policy target is the improved policy pi' = softmax(logits + sigma(completed_q)).

    mode 'train': playout cap randomisation (KataGo). Each decision is a full search with probability
        mcts_full_search_prob (num_mcts_simulation_per_step simulations), otherwise a fast search
        (mcts_fast_num_simulation). Only full searches are recorded as training samples: their policy
        targets are good, and the fast moves make games several times cheaper. Gumbel noise
        (mcts_gumbel_scale) samples the played moves.
    mode 'eval': every decision is a full search (eval_num_mcts_simulation_per_step), greedy (gumbel_scale 0).
    Returns None when interrupted.
    """
    is_train = mode == 'train'
    env = SplendorRLEnv(max_turns=params['max_game_turns'], discard_over_limit=params.get('discard_over_limit', True))
    env.reset(derive_seed(seed, SEED_DEAL))

    if is_train:
        full_prob = params.get('mcts_full_search_prob', 1.)
        full_search = (params['num_mcts_simulation_per_step'], params['mcts_max_num_considered_actions'])
        fast_search = (params.get('mcts_fast_num_simulation', params['num_mcts_simulation_per_step']),
                       params.get('mcts_fast_max_num_considered_actions', params['mcts_max_num_considered_actions']))
    else:
        full_prob = 1.
        full_search = fast_search = (params['eval_num_mcts_simulation_per_step'],
                                     params.get('eval_mcts_max_num_considered_actions', params['mcts_max_num_considered_actions']))
    cap_rng = random.Random(derive_seed(seed, SEED_PLAYOUT_CAP))
    observations, masks, policies, seats = list(), list(), list(), list()
    num_decisions = num_full = 0
    t0 = time.time()
    while not env.terminated:
        seat = env.acting_player
        full = full_prob >= 1. or cap_rng.random() < full_prob
        n_simulation, n_considered = full_search if full else fast_search
        mcts = SplendorMCTS(N_ACTIONS,
                            predictor=None,
                            n_simulation=n_simulation,
                            max_num_considered_actions=n_considered,
                            gumbel_scale=params['mcts_gumbel_scale'] if is_train else 0.,
                            value_scale=params['mcts_value_scale'],
                            maxvisit_init=params['mcts_maxvisit_init'],
                            rescale_values=params['mcts_rescale_values'],
                            rollout_lambda=params['mcts_rollout_lambda'],
                            rollout_policy=params['mcts_rollout_policy'],
                            max_search_depth=params['mcts_max_search_depth'],
                            interrupt_callback=should_stop,
                            seed=derive_seed(seed, SEED_SEARCH, num_decisions))
        record = is_train and full
        if record:
            observations.append(env.observation_bytes())
            masks.append(env.legal_mask_bytes())
            seats.append(seat)
        search_result = yield from mcts.simulate_iter(env, seat_models[seat])
        if search_result is None:
            return None
        action_bin, policy_target = search_result
        if record:
            policies.append(policy_target.astype(np.float32))
        env.step(action_bin)
        num_decisions += 1
        num_full += full

    # a truncated game has no result: rewards all zero and value weight 0 (its positions train the policy only)
    rewards, value_weight = env.value_target()
    result = {
        'game_id': game_id,
        'seed': seed,
        'mode': mode,
        'rewards': rewards,
        'end_reason': env.end_reason,
        'turns': env.game.turn,
        'passes': env.game.passes,
        'points': [p.points for p in env.game.players],
        'seconds': time.time() - t0,
        'decisions': num_decisions,
        'full_searches': num_full,
    }
    if is_train:
        # recorded (full search) decisions; final rewards relative to each decision's acting player,
        # the order of the value head's outputs
        n = len(observations)
        result['samples'] = {
            'obs': np.frombuffer(b''.join(observations), dtype=np.uint8).reshape(n, N_OBSERVATION).copy(),
            'probs': np.stack(policies) if n else np.zeros((0, N_ACTIONS), dtype=np.float32),
            'mask': np.frombuffer(b''.join(masks), dtype=np.uint8).reshape(n, N_ACTIONS).astype(bool),
            'values': (np.stack([to_relative(rewards, s) for s in seats]).astype(np.float32) if n
                       else np.zeros((0, N_PLAYERS), dtype=np.float32)),
            'value_weight': np.full(n, value_weight, dtype=np.float32),
        }
    return result


def play_game(game_id, seed, seat_slots, predictor_by_slot, params, mode, stop_event=None):
    """play_game_iter driven in this process by one predictor(request bytes) -> reply bytes per model slot"""
    def should_stop():
        return interrupt.interrupt_callback() or (stop_event is not None and stop_event.is_set())

    game = play_game_iter(game_id, seed, list(seat_slots), params, mode, should_stop)
    try:
        slot, request = next(game)
        while True:
            slot, request = game.send(predictor_by_slot[slot](request))
    except StopIteration as stop:
        return stop.value


# ============================================================================ process lifetime
# Children are daemons, but a daemon is only cleaned up when its parent exits normally. If the parent is
# killed (SIGKILL, OOM), the children are re-parented and would wait forever, so every long-running child
# loop also stops when its parent is gone.
_PARENT_PID = None


def _remember_parent():
    global _PARENT_PID
    _PARENT_PID = os.getppid()


def _parent_alive():
    return _PARENT_PID is None or os.getppid() == _PARENT_PID


def parse_cpu_list(spec):
    """None / '' -> None; [0, 2, 3] or '0,2-23' -> sorted list of CPU ids"""
    if spec is None or spec == '':
        return None
    if isinstance(spec, (list, tuple)):
        return sorted(int(c) for c in spec)
    cpus = set()
    for part in str(spec).split(','):
        part = part.strip()
        if '-' in part:
            lo, hi = part.split('-')
            cpus.update(range(int(lo), int(hi) + 1))
        elif part:
            cpus.add(int(part))
    return sorted(cpus)


def _set_cpu_affinity(spec, name):
    cpus = parse_cpu_list(spec)
    if cpus is None or not hasattr(os, 'sched_setaffinity'):
        return
    try:
        os.sched_setaffinity(0, cpus)
    except OSError as e:
        logging.warning(f'{name}: cannot set CPU affinity {cpus}: {e}')


# ============================================================================ wire format
# Requests and replies travel as raw bytes (Connection.send_bytes / recv_bytes), no pickling:
#   request = observation as uint8 (every vocab is <= 256) + legal mask as uint8   -> N_OBSERVATION + N_ACTIONS bytes
#   reply   = action probabilities + values as float32                              -> (N_ACTIONS + N_PLAYERS) * 4 bytes
assert max(VOCAB_SIZES.values()) <= 256, 'observation values must fit in uint8'
REQUEST_BYTES = N_OBSERVATION + N_ACTIONS


def encode_request(observation, mask):
    return np.concatenate((observation.astype(np.uint8), mask.astype(np.uint8))).tobytes()


def decode_requests(blobs):
    """list of request bytes -> (observations uint8 [B, N_OBSERVATION], masks bool [B, N_ACTIONS])"""
    batch = np.frombuffer(bytearray(b''.join(blobs)), dtype=np.uint8).reshape(len(blobs), REQUEST_BYTES)
    return batch[:, :N_OBSERVATION], batch[:, N_OBSERVATION:].view(bool)


def decode_reply(blob):
    reply = np.frombuffer(blob, dtype=np.float32)
    return reply[:N_ACTIONS], reply[N_ACTIONS:]


def make_symmetry_tables(num, seed):
    """num random symmetries of the encoding (splendor.rl_env.SYMMETRY_GROUPS) as index tables:
    request_index [num, REQUEST_BYTES]: transformed request = request[request_index[k]];
    reply_index [num, N_ACTIONS + N_PLAYERS]: reply in the original action order = transformed reply[reply_index[k]]"""
    rng = np.random.default_rng(seed)
    request_index = np.empty((num, REQUEST_BYTES), dtype=np.int16)
    reply_index = np.empty((num, N_ACTIONS + N_PLAYERS), dtype=np.int16)
    for k in range(num):
        obs_index, action_index = random_symmetry_indices(rng)
        request_index[k] = np.concatenate((obs_index, N_OBSERVATION + action_index))
        reply_index[k] = np.concatenate((np.argsort(action_index), N_ACTIONS + np.arange(N_PLAYERS)))
    return request_index, reply_index


def with_random_symmetry(predictor, tables, rng):
    """AlphaGo Zero's evaluation of a randomly transformed position: the network sees the state with its
    interchangeable slots permuted at random, the policy is mapped back to the original action order."""
    request_index, reply_index = tables
    num = len(request_index)

    def symmetric_predictor(request):
        k = rng.randrange(num)
        reply = predictor(np.frombuffer(request, dtype=np.uint8)[request_index[k]].tobytes())
        return np.frombuffer(reply, dtype=np.float32)[reply_index[k]].tobytes()
    return symmetric_predictor


def make_local_predictor(model):
    """predictor running the model in this process (num_game_process == 0)"""
    def predictor(request):
        observations, masks = decode_requests([request])
        probs, values = model.predict_batch(observations, masks)
        return np.concatenate((probs[0], values[0])).astype(np.float32).tobytes()
    return predictor


# ============================================================================ transport
# Game worker <-> predict process, one channel per game worker, created by GamePool:
#   requests [groups, group_size, REQUEST_BYTES] uint8   shared memory (multiprocessing RawArray)
#   models   [groups, group_size] uint8                  model index of each request (evaluation: best / new)
#   replies  [groups, group_size, REPLY_FLOATS] float32
#   doorbell: a pipe; the worker keeps one end, the predict process the other, nobody else (so either side
#             sees EOF when the other one dies).
# A worker writes the n requests of one of its groups into rows 0..n-1 and sends (group, n); the predict
# process evaluates them with other blocks in one batch, writes the n replies and sends (group). A group's rows
# belong to exactly one side at a time. One small message per block of requests instead of one per request.
REPLY_FLOATS = N_ACTIONS + N_PLAYERS
_DOORBELL_REQUEST = struct.Struct('<HH')
_DOORBELL_REPLY = struct.Struct('<H')


def make_channel(ctx, num_groups, group_size):
    """-> (arrays, worker_end, predict_end); arrays is picklable as a process argument"""
    arrays = (num_groups, group_size,
              ctx.RawArray('B', num_groups * group_size * REQUEST_BYTES),
              ctx.RawArray('B', num_groups * group_size),
              ctx.RawArray('f', num_groups * group_size * REPLY_FLOATS))
    worker_end, predict_end = ctx.Pipe(duplex=True)
    return arrays, worker_end, predict_end


def channel_views(arrays):
    num_groups, group_size, requests, models, replies = arrays
    return (np.frombuffer(requests, dtype=np.uint8).reshape(num_groups, group_size, REQUEST_BYTES),
            np.frombuffer(models, dtype=np.uint8).reshape(num_groups, group_size),
            np.frombuffer(replies, dtype=np.float32).reshape(num_groups, group_size, REPLY_FLOATS))


# ============================================================================ predict process
def predict_batch_process(server_id, model_param_dict, state_dict_list, channels, control_queue, ack_queue, predict_batch_size, predict_batch_wait_seconds, stop_event, log_level, log_interval=60., cpu_affinity=None, max_in_flight=2, early_launch_rows=128, log_file=None):
    """Batched inference, PyTorch only. One process hosts every model slot of the experiment (SLOT_CURRENT:
    the latest weights for self-play; SLOT_BEST / SLOT_NEW: evaluation), because several processes sharing
    one GPU time-slice it and are many times slower than a single process. It lives as long as the experiment.

    control_queue: ('weights', slot, state_dict, version) replaces a slot's weights in place (the CUDA graphs
    read the parameter tensors, so they pick the new weights up); batches queued before keep the old ones
    (same CUDA stream). Each update is acknowledged on ack_queue with (server_id, slot, version).

    channels: [(channel arrays, doorbell end)] of the game workers this process serves (see "transport").
    Blocks of requests that arrive together form a batch (up to predict_batch_size requests; whole blocks),
    which is split by model for the forward passes. Pipelined: while the GPU computes batch k this loop
    collects batch k+1 and queues it behind k (up to max_in_flight batches) once early_launch_rows requests
    wait, so the GPU keeps computing while k's results are written and its doorbells rung.
    """
    max_in_flight = max(1, min(int(max_in_flight), PREDICTION_RING - 1))
    _remember_parent()
    init_logging(log_level, log_file)
    _set_cpu_affinity(cpu_affinity, f'predict_batch_process {server_id}')
    model_param_dict = dict(model_param_dict, transition_buffer_len=1)
    models = list()
    for state_dict in state_dict_list:
        model = SplendorAlphaGoZero(**model_param_dict)
        model.load_state_dict(state_dict)
        model.eval()
        if model_param_dict.get('inference_cuda_graph_batch_sizes'):
            model.enable_cuda_graphs(model_param_dict['inference_cuda_graph_batch_sizes'])
        models.append(model)
    workers = [(conn,) + channel_views(arrays) for arrays, conn in channels]
    logging.info(f'predict_batch_process {server_id} pid {os.getpid()} started on {models[0].device}: '
                 f'{len(models)} model(s), serving {len(workers)} game workers')

    selector = selectors.DefaultSelector()
    for w, (conn, _, _, _) in enumerate(workers):
        selector.register(conn, selectors.EVENT_READ, data=w)
    num_open = len(workers)

    def receive(timeout, blocks, total):
        """read every queued message of the doorbells that are ready (a worker can have one block in flight
        per group); -> number of requests added to blocks"""
        nonlocal num_open
        added = 0
        for key, _ in selector.select(timeout):
            conn = key.fileobj
            while total + added < predict_batch_size:
                try:
                    group, n = _DOORBELL_REQUEST.unpack(conn.recv_bytes())
                except (EOFError, OSError):
                    selector.unregister(conn)
                    num_open -= 1
                    break
                blocks.append((key.data, group, n))
                added += n
                if not conn.poll(0):
                    break
        return added

    def launch(blocks):
        """queue the forward passes; -> (blocks, [(rows or None, PendingPrediction)], batch size)"""
        batch = np.concatenate([workers[w][1][g, :n] for w, g, n in blocks])
        model_ids = np.concatenate([workers[w][2][g, :n] for w, g, n in blocks])
        observations, masks = batch[:, :N_OBSERVATION], batch[:, N_OBSERVATION:].view(bool)
        if len(models) == 1 or (model_ids == model_ids[0]).all():
            return blocks, [(None, models[int(model_ids[0])].predict_batch_async(observations, masks))], len(batch)
        parts = list()
        for model_id in np.unique(model_ids):
            rows = np.flatnonzero(model_ids == model_id)
            parts.append((rows, models[int(model_id)].predict_batch_async(observations[rows], masks[rows])))
        return blocks, parts, len(batch)

    def finish(pending):
        """wait for the GPU and write the replies into the workers' shared memory"""
        blocks, parts, batch_size = pending
        out = np.empty((batch_size, REPLY_FLOATS), dtype=np.float32)
        for rows, handle in parts:
            probs, values = handle.result()
            if rows is None:
                out[:, :N_ACTIONS], out[:, N_ACTIONS:] = probs, values
            else:
                out[rows, :N_ACTIONS], out[rows, N_ACTIONS:] = probs, values
        offset = 0
        for w, g, n in blocks:
            workers[w][3][g, :n] = out[offset:offset + n]
            offset += n
        return blocks, batch_size

    def ring_doorbells(done):
        for w, g, n in done[0]:
            try:
                workers[w][0].send_bytes(_DOORBELL_REPLY.pack(g))
            except (BrokenPipeError, OSError):
                pass   # the worker is gone; its doorbell reports EOF and is dropped when read
        stats['predicted'] += done[1]
        stats['blocks'] += len(done[0])
        stats['batches'] += 1

    pendings = collections.deque()       # launched batches, oldest first (they run in order on one stream)
    blocks, total = list(), 0            # requests collected for the next batch
    stats = {'predicted': 0, 'batches': 0, 'blocks': 0, 'early': 0, 'idle': 0., 'busy': 0., 'finish': 0., 'reply': 0., 'launch': 0.}
    last_log_time, last_cpu_time = time.time(), time.process_time()
    def apply_control_messages():
        while True:
            try:
                kind, slot, state_dict, version = control_queue.get_nowait()
            except Empty:
                return
            assert kind == 'weights', kind
            models[slot].load_state_dict(state_dict)
            ack_queue.put((server_id, slot, version))
            logging.debug(f'predict_batch_process {server_id}: slot {slot} weights version {version}')

    # stop_event is a multiprocessing.Event: is_set() takes a lock shared with every game process, so it is
    # checked a few times per second, not once per batch
    next_stop_check = next_control_check = 0.
    while num_open > 0:
        now = time.time()
        if now >= next_control_check:
            apply_control_messages()
            next_control_check = now + 0.02
        if now >= next_stop_check:
            if stop_event.is_set() or interrupt.interrupt_callback() or not _parent_alive():
                break
            next_stop_check = now + 0.2
        if now - last_log_time > log_interval:
            elapsed = now - last_log_time
            cpu_time = time.process_time()
            n, nb = stats['predicted'], max(stats['batches'], 1)
            logging.info(f'predict_batch_process {server_id}: {n / elapsed:.0f} predictions/s, '
                         f'mean batch {n / nb:.1f} ({stats["blocks"] / nb:.1f} blocks), {100 * stats["early"] / nb:.0f}% launched early, '
                         f'GPU idle {100 * stats["idle"] / elapsed:.0f}% of the time; per batch: busy-wait {1000 * stats["busy"] / nb:.2f} ms, '
                         f'finish {1000 * stats["finish"] / nb:.2f} ms, reply {1000 * stats["reply"] / nb:.2f} ms, '
                         f'launch {1000 * stats["launch"] / nb:.2f} ms, process CPU {100 * (cpu_time - last_cpu_time) / elapsed:.0f}%')
            stats.update(predicted=0, batches=0, blocks=0, early=0, idle=0., busy=0., finish=0., reply=0., launch=0.)
            last_log_time, last_cpu_time = now, cpu_time

        if not pendings:
            # GPU idle: block for the first doorbell, then wait up to predict_batch_wait_seconds for more
            idle_begin = time.time()
            if not blocks:
                total += receive(0.05, blocks, total)
                if not blocks:
                    stats['idle'] += time.time() - idle_begin
                    continue
            deadline = time.time() + predict_batch_wait_seconds
            while total < predict_batch_size:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                added = receive(remaining, blocks, total)
                if not added:
                    break
                total += added
            launch_begin = time.time()
            pendings.append(launch(blocks))
            blocks, total = list(), 0
            stats['idle'] += launch_begin - idle_begin
            stats['launch'] += time.time() - launch_begin
            continue

        # GPU busy: take whatever arrives until the oldest batch is done (non-blocking polls; this process has
        # a core). With fewer than max_in_flight batches queued, the next one is launched as soon as
        # early_launch_rows requests wait, so the GPU does not idle while this process takes results, writes
        # the replies and launches the next batch
        busy_begin = time.time()
        oldest = pendings[0]
        while not all(h.ready() for _, h in oldest[1]):
            if total < predict_batch_size:
                total += receive(0, blocks, total)
            if len(pendings) < max_in_flight and total >= early_launch_rows:
                launch_begin = time.time()
                pendings.append(launch(blocks))
                blocks, total = list(), 0
                stats['early'] += 1
                stats['launch'] += time.time() - launch_begin
                busy_begin += time.time() - launch_begin
        finish_begin = time.time()
        stats['busy'] += finish_begin - busy_begin
        done = finish(pendings.popleft())
        finished = time.time()
        stats['finish'] += finished - finish_begin
        if not pendings and blocks:
            # nothing queued on the GPU: launch first, then answer the workers
            pendings.append(launch(blocks))
            blocks, total = list(), 0
            launched = time.time()
            ring_doorbells(done)
            stats['launch'] += launched - finished
            stats['reply'] += time.time() - launched
        else:
            ring_doorbells(done)
            stats['reply'] += time.time() - finished
            if blocks and len(pendings) < max_in_flight:
                launch_begin = time.time()
                pendings.append(launch(blocks))
                blocks, total = list(), 0
                stats['launch'] += time.time() - launch_begin
    logging.info(f'predict_batch_process {server_id} exit')


# ============================================================================ game worker process
def game_worker_process(worker_id, params, task_queue, result_queue, channel, doorbell, stop_event, log_level, log_file=None):
    """Plays its games in ONE thread: num_game_groups_per_process groups of num_games_per_group games.

    The games of a group run until each needs a network evaluation; the group's requests go to the predict
    process as one block (see "transport"); while it is evaluated, the worker advances another group. No GIL
    contention between game threads and no per-request pipe messages. A game that raises is reported as a
    failed result, so nobody waits for it. Tasks are (game_id, seed, seat_slots, mode, tag); the worker lives
    as long as the game pool (until stop_event, or a None task).
    """
    _remember_parent()
    init_logging(log_level, log_file)
    _set_cpu_affinity(params.get('game_process_cpu_affinity'), f'game_worker_process {worker_id}')
    requests, model_rows, replies = channel_views(channel)
    num_groups, group_size = requests.shape[0], requests.shape[1]
    games = [[None] * group_size for _ in range(num_groups)]       # running game generators
    meta = [[None] * group_size for _ in range(num_groups)]        # (game_id, seat_slots, mode, tag)
    rows = [list() for _ in range(num_groups)]                     # slot of each request row of the block in flight
    sent_at = [0.] * num_groups
    in_flight = [False] * num_groups
    stop = [False]
    stash = list()               # a task taken while waiting idle
    no_more_tasks = [False]
    # the same tables in every worker; which one evaluates a request is drawn from the game's own RNG
    symmetry = make_symmetry_tables(1024, seed=derive_seed(params['seed'], SEED_SYMMETRY)) if params.get('mcts_random_symmetry', False) else None
    symmetry_rngs = [[None] * group_size for _ in range(num_groups)]
    symmetry_used = [[0] * group_size for _ in range(num_groups)]
    stats = {'predicted': 0, 'blocks': 0, 'wait': 0.}

    def should_stop():
        return stop[0]

    def next_task(block=False):
        if stash:
            return stash.pop()
        if no_more_tasks[0]:
            return None
        try:
            task = task_queue.get(block=True, timeout=0.2) if block else task_queue.get_nowait()
        except Empty:
            return None
        if task is None:
            no_more_tasks[0] = True
        return task

    def advance(g, s, reply):
        """start (reply None) or resume game (g, s); -> (model_id, request) or None when the game is over"""
        try:
            return next(games[g][s]) if reply is None else games[g][s].send(reply)
        except StopIteration as stop_iteration:
            result = stop_iteration.value
            if result is not None:
                result['seat_server_ids'] = meta[g][s][1]
                result['tag'] = meta[g][s][3]
                result_queue.put(result)
        except Exception:
            logging.exception(f'game_worker {worker_id}: game {meta[g][s][0]} failed')
            result_queue.put({'game_id': meta[g][s][0], 'seat_server_ids': meta[g][s][1], 'mode': meta[g][s][2],
                              'tag': meta[g][s][3], 'failed': True})
        games[g][s] = None
        return None

    def run_group(g, has_replies):
        """resume the games of group g with the replies of its block, start new games in free slots, send the next block"""
        reply_of_slot = dict()
        if has_replies:
            for r, s in enumerate(rows[g]):
                reply = replies[g, r]
                if symmetry is not None:
                    reply = reply[symmetry[1][symmetry_used[g][s]]]
                reply_of_slot[s] = reply.tobytes()
            stats['predicted'] += len(rows[g])
            stats['blocks'] += 1
            stats['wait'] += time.time() - sent_at[g]
        n = 0
        new_rows = list()
        for s in range(group_size):
            out = advance(g, s, reply_of_slot[s]) if games[g][s] is not None else None
            while out is None and not stop[0]:
                task = next_task()
                if task is None:
                    break
                game_id, seed, seat_slots, mode, tag = task
                games[g][s] = play_game_iter(game_id, seed, seat_slots, params, mode, should_stop)
                meta[g][s] = (game_id, seat_slots, mode, tag)
                if symmetry is not None:
                    symmetry_rngs[g][s] = random.Random(derive_seed(seed, SEED_SYMMETRY))
                out = advance(g, s, None)
            if out is None:
                continue
            model_id, request = out
            request = np.frombuffer(request, dtype=np.uint8)
            if symmetry is not None:
                k = symmetry_rngs[g][s].randrange(len(symmetry[0]))
                symmetry_used[g][s] = k
                request = request[symmetry[0][k]]
            requests[g, n] = request
            model_rows[g, n] = model_id
            new_rows.append(s)
            n += 1
        rows[g] = new_rows
        in_flight[g] = n > 0
        if n:
            sent_at[g] = time.time()
            doorbell.send_bytes(_DOORBELL_REQUEST.pack(g, n))

    log_interval = params.get('stats_log_interval', 60.)
    last_log_time, last_cpu_time = time.time(), time.process_time()
    next_stop_check = 0.
    finished = False
    for g in range(num_groups):
        run_group(g, False)
    while True:
        now = time.time()
        if now >= next_stop_check:
            # stop_event is a multiprocessing.Event: is_set() takes a lock shared by all processes
            if stop_event.is_set() or interrupt.interrupt_callback() or not _parent_alive():
                stop[0] = True
                break
            next_stop_check = now + 0.2
        if now - last_log_time > log_interval:
            n = stats['predicted']
            logging.info(f'game_worker {worker_id}: process CPU {100 * (time.process_time() - last_cpu_time) / (now - last_log_time):.0f}%, '
                         f'{n / (now - last_log_time):.0f} predictions/s, mean block {n / max(stats["blocks"], 1):.1f}, '
                         f'mean block round trip {1000 * stats["wait"] / max(stats["blocks"], 1):.2f} ms')
            stats.update(predicted=0, blocks=0, wait=0.)
            last_log_time, last_cpu_time = now, time.process_time()
        if not any(in_flight):
            if no_more_tasks[0] and not stash:
                finished = True
                break                                  # every game of this worker is done
            task = next_task(block=True)               # idle: wait for work
            if task is not None:
                stash.append(task)
                for g in range(num_groups):
                    run_group(g, False)
            continue
        try:
            if not doorbell.poll(0.2):
                continue
            group, = _DOORBELL_REPLY.unpack(doorbell.recv_bytes())
        except (EOFError, OSError):
            logging.error(f'game_worker {worker_id}: the predict process is gone')
            break
        run_group(group, True)
        if not no_more_tasks[0] or stash:
            for g in range(num_groups):               # a group that ran out of tasks earlier may get new ones
                if not in_flight[g]:
                    run_group(g, False)
    if stop[0] or not finished:
        # stopped early: the parent may have stopped reading, and results still buffered in the queue's
        # feeder thread must not block this process's exit
        result_queue.cancel_join_thread()
    # otherwise the exit flushes the buffered results (the parent reads until it has every result)
    logging.info(f'game_worker_process {worker_id} exit')


# ============================================================================ game pool
class GamePool:
    """Game workers and predict process(es), started once and kept for the whole experiment.

    The predict process hosts NUM_SLOTS model slots; set_weights(slot, ...) replaces one (and waits until
    every predict process has loaded it, so games submitted afterwards never see the old weights).
    submit() queues tasks (game_id, seed, seat_slots, mode, tag); get_result() returns finished games in
    completion order (a failed game comes back as {'failed': True, ...}). Workers take a new task as soon as
    one of their game slots is free, so new games start while older ones are still running.
    num_predict_process predict processes (default 1) each host every slot; game worker w is served by
    predict process w % num_predict_process over its shared-memory channel (see "transport").
    """

    def __init__(self, params, state_dicts, log_level):
        assert len(state_dicts) == NUM_SLOTS
        ctx = mp.get_context('spawn')
        num_game_process = params['num_game_process']
        num_groups = params.get('num_game_groups_per_process', 2)
        group_size = params.get('num_games_per_group', 12)
        num_predict_process = max(1, min(params.get('num_predict_process', 1), num_game_process))
        log_file = params.get('_log_file')
        self.capacity = num_game_process * num_groups * group_size   # games played at the same time
        self.outstanding = {'train': 0, 'eval': 0}
        self.stop_event = ctx.Event()
        self.task_queue = ctx.Queue()
        self.result_queue = ctx.Queue()
        self.ack_queue = ctx.Queue()
        self.control_queues = [ctx.Queue() for _ in range(num_predict_process)]
        channels = [make_channel(ctx, num_groups, group_size) for _ in range(num_game_process)]
        model_param_dict = params['model_param_dict']

        self.predict_processes, self.worker_processes = list(), list()
        for k in range(num_predict_process):
            served = [(arrays, predict_end) for w, (arrays, _, predict_end) in enumerate(channels) if w % num_predict_process == k]
            p = ctx.Process(target=predict_batch_process,
                            args=(str(k), model_param_dict, state_dicts, served, self.control_queues[k], self.ack_queue,
                                  params['predict_batch_size'], params['predict_batch_wait_seconds'], self.stop_event,
                                  log_level, params.get('stats_log_interval', 60.), params.get('predict_process_cpu_affinity'),
                                  params.get('predict_max_in_flight', 2), params.get('predict_early_launch_rows', 128), log_file),
                            daemon=True)
            p.start()
            self.predict_processes.append(p)
        for worker_id, (arrays, worker_end, _) in enumerate(channels):
            p = ctx.Process(target=game_worker_process,
                            args=(worker_id, params, self.task_queue, self.result_queue, arrays, worker_end,
                                  self.stop_event, log_level, log_file),
                            daemon=True)
            p.start()
            self.worker_processes.append(p)
        # every doorbell end now lives in exactly one child: when a child dies, its peer sees EOF
        for _, worker_end, predict_end in channels:
            worker_end.close()
            predict_end.close()
        logging.info(f'game pool started: {num_game_process} game workers x {num_groups} x {group_size} games, '
                     f'{num_predict_process} predict process(es) with {NUM_SLOTS} model slots')

    def submit(self, tasks):
        for task in tasks:
            self.task_queue.put(task)
            self.outstanding[task[3]] += 1

    def set_weights(self, slot, state_dict, version, timeout=300.):
        for queue in self.control_queues:
            queue.put(('weights', slot, state_dict, version))
        pending = len(self.control_queues)
        deadline = time.time() + timeout
        while pending:
            try:
                _, ack_slot, ack_version = self.ack_queue.get(timeout=1.)
            except Empty:
                self.check_alive()
                if time.time() > deadline:
                    raise RuntimeError(f'predict process(es) did not load slot {slot} weights within {timeout:.0f}s')
                continue
            if (ack_slot, ack_version) == (slot, version):
                pending -= 1

    def get_result(self, timeout=1.):
        """-> the next finished game, or None after timeout (raises RuntimeError if a pool process died)"""
        try:
            result = self.result_queue.get(block=True, timeout=timeout)
        except Empty:
            self.check_alive()
            return None
        self.outstanding[result['mode']] -= 1
        return result

    def check_alive(self):
        dead = [p.pid for p in self.predict_processes + self.worker_processes if not p.is_alive()]
        if dead:
            raise RuntimeError(f'game pool process(es) {dead} exited (predict exit codes '
                               f'{[p.exitcode for p in self.predict_processes]}, worker exit codes '
                               f'{[p.exitcode for p in self.worker_processes]}); their games are lost')

    def close(self):
        self.stop_event.set()
        # queued tasks / weights nobody will read must not block this process's exit
        self.task_queue.cancel_join_thread()
        for queue in self.control_queues:
            queue.cancel_join_thread()
        for p in self.predict_processes + self.worker_processes:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
        logging.info('game pool stopped')


class LocalGamePool:
    """GamePool's interface, playing one game at a time in this process (num_game_process == 0: debugging,
    no GPU). get_result() plays the oldest queued task."""

    def __init__(self, params, state_dicts, log_level):
        model_param_dict = dict(params['model_param_dict'], transition_buffer_len=1)
        self.params = params
        self.models = list()
        for state_dict in state_dicts:
            model = SplendorAlphaGoZero(**model_param_dict)
            model.load_state_dict(state_dict)
            self.models.append(model)
        self.capacity = 1
        self.outstanding = {'train': 0, 'eval': 0}
        self.tasks = collections.deque()

    def submit(self, tasks):
        for task in tasks:
            self.tasks.append(task)
            self.outstanding[task[3]] += 1

    def set_weights(self, slot, state_dict, version, timeout=None):
        self.models[slot].load_state_dict(state_dict)

    def get_result(self, timeout=1.):
        if not self.tasks:
            time.sleep(min(timeout, 0.1))
            return None
        game_id, seed, seat_slots, mode, tag = self.tasks.popleft()
        predictors = {slot: make_local_predictor(self.models[slot]) for slot in set(seat_slots)}
        if self.params.get('mcts_random_symmetry', False):
            tables = make_symmetry_tables(64, seed=derive_seed(self.params['seed'], SEED_SYMMETRY))
            rng = random.Random(derive_seed(seed, SEED_SYMMETRY))
            predictors = {slot: with_random_symmetry(p, tables, rng) for slot, p in predictors.items()}
        self.outstanding[mode] -= 1
        try:
            result = play_game(game_id, seed, seat_slots, predictors, self.params, mode)
        except Exception:
            logging.exception(f'game {game_id} failed')
            return {'game_id': game_id, 'seat_server_ids': seat_slots, 'mode': mode, 'tag': tag, 'failed': True}
        if result is None:   # interrupted
            return None
        result['seat_server_ids'] = seat_slots
        result['tag'] = tag
        return result

    def check_alive(self):
        pass

    def close(self):
        pass


def make_game_pool(params, state_dicts, log_level):
    if params['num_game_process'] <= 0:
        return LocalGamePool(params, state_dicts, log_level)
    return GamePool(params, state_dicts, log_level)


def _ensure_fd_limit(num_fds_needed):
    """每条 pipe 占两个文件描述符，推理进程还会各持有一份全部写端；macOS 默认软上限只有 256。"""
    try:
        import resource
    except ImportError:  # Windows
        return
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft == resource.RLIM_INFINITY or soft >= num_fds_needed:
        return
    target = num_fds_needed if hard == resource.RLIM_INFINITY else min(num_fds_needed, hard)
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
        logging.info(f'raised open file soft limit {soft} -> {target}')
    except (ValueError, OSError) as e:
        logging.warning(f'cannot raise open file limit ({soft}, need ~{num_fds_needed}): {e}')


def _log_game(mode, result, num_finished, num_total, begin_time):
    logging.info(f'[{mode}] game {result["game_id"]} finished ({num_finished}/{num_total}): end_reason={result["end_reason"]}, '
                 f'turns={result["turns"]}, passes={result.get("passes", 0)}, points={result["points"]}, seats={result["seat_server_ids"]}, '
                 f'game cost {result["seconds"]:.1f}s, total {time.time() - begin_time:.1f}s')


def summarize_games(results):
    reasons = dict()
    for r in results:
        reasons[r['end_reason']] = reasons.get(r['end_reason'], 0) + 1
    avg_turns = float(np.mean([r['turns'] for r in results])) if results else 0.
    return reasons, avg_turns


# ============================================================================ experiment directory
def experiment_paths(experiment_dir):
    checkpoints = os.path.join(experiment_dir, 'checkpoints')
    return {'dir': experiment_dir,
            'config': os.path.join(experiment_dir, 'config.yml'),
            'data': os.path.join(experiment_dir, 'train_data.txt'),
            'log': os.path.join(experiment_dir, 'train.log'),
            'latest': os.path.join(checkpoints, 'latest.pth'),      # full training state (exact resume)
            'best': os.path.join(checkpoints, 'best.pth'),          # weights of the best model
            'eval': os.path.join(checkpoints, 'eval_new.pth'),      # weights of the model being evaluated
            'archive': os.path.join(checkpoints, 'weights_iteration_%05d.pth')}


def prepare_new_experiment(params, paths):
    """create the experiment directory; refuse one that already holds an experiment"""
    used = [paths[k] for k in ('config', 'data', 'latest') if os.path.exists(paths[k])]
    if used:
        raise RuntimeError(f'{paths["dir"]} already holds an experiment ({", ".join(used)}): resume it with '
                           f'"python train_splendor.py --resume {paths["dir"]}" or choose another experiment_name')
    os.makedirs(os.path.join(paths['dir'], 'checkpoints'), exist_ok=True)
    with open(paths['config'], 'w', encoding='UTF-8') as f:
        yaml.safe_dump({k: v for k, v in params.items() if not k.startswith('_')}, f, allow_unicode=True, sort_keys=True)


def _rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _set_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state.get('cuda') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def _checkpoint_header(params, kind):
    return {'format': CHECKPOINT_FORMAT_VERSION, 'kind': kind,
            'versions': version_info(params.get('discard_over_limit', True)),
            'architecture': architecture_of(params['model_param_dict']),
            'experiment': params['experiment_dir'], 'saved_at': time.strftime('%Y-%m-%d %H:%M:%S')}


def save_weights(params, path, state_dict, kind, **extra):
    save_checkpoint(path, dict(_checkpoint_header(params, kind), model=state_dict, **extra))


def save_training_state(params, paths, model, state):
    """checkpoints/latest.pth: everything an exact resume needs"""
    save_checkpoint(paths['latest'], dict(_checkpoint_header(params, 'training_state'),
                                          model=model.state_dict(), optimizer=model.optimizer.state_dict(),
                                          scheduler=model.scheduler.state_dict(),
                                          model_train_step_num=model.train_step_num,
                                          train_state=dict(state), rng=_rng_state()))


def _check_checkpoint(checkpoint, params, path, kind):
    """strict: same checkpoint format, kind, rules / observation / action versions and architecture"""
    problems = []
    if not isinstance(checkpoint, dict) or checkpoint.get('format') != CHECKPOINT_FORMAT_VERSION:
        found = checkpoint.get('format') if isinstance(checkpoint, dict) else type(checkpoint).__name__
        problems.append(f'checkpoint format {found!r} != {CHECKPOINT_FORMAT_VERSION}')
    else:
        if checkpoint.get('kind') != kind:
            problems.append(f'kind {checkpoint.get("kind")!r} != {kind!r}')
        problems += versions_mismatch(checkpoint.get('versions'), version_info(params.get('discard_over_limit', True)))
        expected = architecture_of(params['model_param_dict'])
        if checkpoint.get('architecture') != expected:
            problems.append(f'architecture {checkpoint.get("architecture")} != {expected}')
    if problems:
        raise ValueError(f'cannot resume from {path}: {"; ".join(problems)}')


def _check_state_keys(state_dict, model, path):
    own = model.state_dict()
    missing = sorted(set(own) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(own))
    mismatched = sorted(k for k in own if k in state_dict and tuple(own[k].shape) != tuple(state_dict[k].shape))
    if missing or unexpected or mismatched:
        raise ValueError(f'cannot resume from {path}: missing {missing[:3]} ({len(missing)}), unexpected '
                         f'{unexpected[:3]} ({len(unexpected)}), other shapes {mismatched[:3]} ({len(mismatched)})')


def resume_experiment(params, paths, model, state):
    """Exact resume from checkpoints/latest.pth (strict checks): network, optimizer, scheduler, counters,
    random states, best model; the data file is cut back to the checkpoint and the replay buffer refilled
    from it. -> best state dict"""
    checkpoint = load_checkpoint(paths['latest'])
    _check_checkpoint(checkpoint, params, paths['latest'], 'training_state')
    _check_state_keys(checkpoint['model'], model, paths['latest'])
    model.load_state_dict(checkpoint['model'], strict=True)
    model.optimizer.load_state_dict(checkpoint['optimizer'])
    model.scheduler.load_state_dict(checkpoint['scheduler'])
    model.train_step_num = checkpoint['model_train_step_num']
    state.update(checkpoint['train_state'])

    best = load_checkpoint(paths['best'])
    _check_checkpoint(best, params, paths['best'], 'best')
    _check_state_keys(best['model'], model, paths['best'])

    if truncate_data_file(paths['data'], state['data_file_bytes']):
        logging.info(f'{paths["data"]}: samples written after the checkpoint removed (cut back to {state["data_file_bytes"]} bytes)')
    samples, stats = read_samples(paths['data'], params.get('discard_over_limit', True),
                                  last=params['model_param_dict'].get('transition_buffer_len', 200000))
    if stats['read'] + stats['invalid'] != state['num_data_written']:
        logging.warning(f'{paths["data"]} holds {stats["read"] + stats["invalid"]} samples, the checkpoint recorded '
                        f'{state["num_data_written"]}')
    for sample in samples:
        model.store_transition(*sample)
    _set_rng_state(checkpoint['rng'])
    logging.info(f'resumed {paths["dir"]} at iteration {state["iteration"]}: {state["train_step_num"]} train steps, '
                 f'next game id {state["next_game_id"]}, replay buffer {model.memory_size()} samples, '
                 f'best model from iteration {best.get("iteration")}')
    return best['model']


def warm_start(model, params):
    """optional: samples of another experiment's data file (same versions) into the replay buffer; mode
    'train' also trains on them with the run's sample reuse, 'fill' only fills the buffer"""
    path = params.get('warm_start_data_path')
    if not path:
        return 0
    samples, stats = read_samples(path, params.get('discard_over_limit', True),
                                  last=params['model_param_dict'].get('transition_buffer_len', 200000))
    train = params.get('warm_start_data_mode', 'fill') == 'train'
    samples_per_step = model.batch_size / params['train_sample_reuse']
    next_train_step, train_steps = params['first_train_data_step'], 0
    for num, sample in enumerate(samples, start=1):
        model.store_transition(*sample)
        while train and num >= next_train_step:
            model.learn()
            train_steps += 1
            next_train_step += samples_per_step
    logging.info(f'warm start from {path}: {len(samples)} samples in the replay buffer, {train_steps} train steps')
    return len(samples)


def _format_metrics(m):
    return (f'value CE {m["value_ce"]:.4f}, value MSE {m["value_mse"]:.4f} (constant prediction {m["value_mse_constant"]:.4f}), '
            f'winner acc {m["winner_acc"]:.3f}, policy CE {m["policy_ce"]:.4f} (target entropy {m["target_entropy"]:.4f}), '
            f'{m["num_samples"]} samples ({m["num_value_samples"]} with a result)')


# ============================================================================ evaluation
def eval_tasks(params, eval_round):
    """new model vs best: each holds two seats; games 2j and 2j+1 share the deal and swap the seats"""
    tasks = list()
    for k in range(params['eval_task_num_games']):
        arrangement = EVAL_SEAT_ARRANGEMENTS[k % len(EVAL_SEAT_ARRANGEMENTS)]
        slots = [SLOT_NEW if a == NEW_MODEL_IDX else SLOT_BEST for a in arrangement]
        tasks.append((k, derive_seed(params['eval_seed'], SEED_EVAL_GAME, k // 2), slots, 'eval', eval_round))
    return tasks


def eval_score(results, num_games):
    """The new model's score over ALL scheduled games: its win share in a game won by points; 0.5 (a draw) for
    every other game: a draw by four passes, a truncated, failed or missing game. completion = games ended by
    the rules (score or draw) / games scheduled. -> dict"""
    total, counts = 0., dict()
    for r in results:
        reason = 'failed' if r.get('failed') else r['end_reason']
        counts[reason] = counts.get(reason, 0) + 1
        if reason == 'score':
            total += sum(reward for reward, slot in zip(r['rewards'], r['seat_server_ids']) if slot == SLOT_NEW)
        else:
            total += 0.5
    missing = num_games - len(results)
    if missing:
        counts['missing'] = missing
    total += 0.5 * missing
    completed = counts.get('score', 0) + counts.get('draw', 0)
    return {'win_rate': total / max(num_games, 1), 'completion': completed / max(num_games, 1), 'counts': counts}


# ============================================================================ train
def train_process(params, n_loop, log_level, resume=False):
    """One experiment in params['experiment_dir']: new (optionally initialised from init_weights_path) or an
    exact resume of an existing one."""
    _remember_parent()
    paths = experiment_paths(params['experiment_dir'])
    for key in REMOVED_PARAMS:
        if key in params:
            logging.warning(f'parameter {key} no longer exists (every experiment writes into its own directory), ignored')
    if not resume:
        prepare_new_experiment(params, paths)
    params['_log_file'] = paths['log']
    init_logging(log_level, paths['log'])
    training_seed = derive_seed(params['seed'], SEED_TRAINING)
    torch.manual_seed(training_seed % (2 ** 63))
    np.random.seed(training_seed % (2 ** 32))
    random.seed(training_seed)

    model_param_dict = params['model_param_dict']
    for key in ('save_train_data', 'default_train_file_path'):
        model_param_dict.pop(key, None)
    model_param_dict['device'] = resolve_device(model_param_dict.get('device'))
    logging.info(f'{"resume" if resume else "new"} experiment {paths["dir"]} (loop {n_loop}), versions '
                 f'{version_info(params.get("discard_over_limit", True))}, params: {params}')

    model = SplendorAlphaGoZero(**model_param_dict)
    state = {'iteration': 0, 'next_game_id': 0, 'train_step_num': 0, 'train_credit': 0., 'num_data': 0,
             'num_data_written': 0, 'data_file_bytes': 0, 'eval_round': 0, 'best_iteration': None}
    if resume:
        best_state_dict = resume_experiment(params, paths, model, state)
    else:
        if params.get('init_weights_path'):
            init_weights_from(model, params['init_weights_path'], strict=params.get('init_weights_strict', True))
        state['num_data'] = warm_start(model, params)
        best_state_dict = get_state_dict_from_model(model)
        save_weights(params, paths['best'], best_state_dict, 'best', iteration=None, eval_round=0)
    writer = TrainDataWriter(paths['data'], params.get('discard_over_limit', True))
    state['data_file_bytes'] = writer.flush()
    if not resume:
        save_training_state(params, paths, model, state)

    pool = make_game_pool(params, [get_state_dict_from_model(model), best_state_dict, best_state_dict], log_level)
    try:
        _train_loop(params, paths, model, state, best_state_dict, pool, writer, n_loop)
    except Exception:
        logging.exception(f'loop {n_loop}: training stopped by an error (resume with --resume {paths["dir"]})')
        raise
    finally:
        pool.close()
        writer.close()
        model.close()


def _train_loop(params, paths, model, state, best_state_dict, pool, writer, n_loop):
    games_per_iteration = params['num_self_play_games_per_iteration']
    samples_per_game = params['num_samples_per_game']
    sample_reuse = params['train_sample_reuse']
    # self-play games submitted but not finished: every game slot busy plus a short queue, so a worker never
    # waits for work and evaluation tasks (queued behind) start soon
    target_outstanding = pool.capacity + (params.get('self_play_queue_margin', 64) if pool.capacity > 1 else 0)
    eval_job = None
    iteration_results = list()
    num_failed = 0
    iteration_begin = run_begin = time.time()

    def finish_eval(job):
        nonlocal best_state_dict
        score = eval_score(job['results'], job['num_games'])
        promote = (score['completion'] >= params.get('eval_min_completion_rate', 0.9)
                   and score['win_rate'] > params['update_model_win_rate_thres'])
        logging.info(f'Evaluation {job["round"]} (model of iteration {job["iteration"]}) finished in '
                     f'{time.time() - job["begin"]:.0f}s: new_model_win_rate={score["win_rate"]:.4f} over all '
                     f'{job["num_games"]} games (non-score games count as draws), completion {score["completion"]:.3f} '
                     f'(minimum {params.get("eval_min_completion_rate", 0.9)}), end_reasons={score["counts"]}')
        if promote:
            best_state_dict = job['state_dict']
            pool.set_weights(SLOT_BEST, best_state_dict, version=job['round'])
            save_weights(params, paths['best'], best_state_dict, 'best', iteration=job['iteration'],
                         eval_round=job['round'], win_rate=score['win_rate'])
            state['best_iteration'] = job['iteration']
            logging.info(f'loop {n_loop}: model of iteration {job["iteration"]} promoted to best ({paths["best"]})')
        else:
            logging.info(f'loop {n_loop}: best model kept (iteration {state["best_iteration"]})')

    while state['iteration'] < params['num_iterations']:
        if interrupt.interrupt_callback() or not _parent_alive():
            logging.info(f'loop {n_loop}: interrupted')
            return
        # -------- keep the pool busy: new games start while older ones are still running
        missing = target_outstanding - pool.outstanding['train']
        if missing > 0:
            tasks = list()
            for _ in range(missing):
                game_id = state['next_game_id']
                state['next_game_id'] += 1
                tasks.append((game_id, derive_seed(params['seed'], SEED_TRAIN_GAME, game_id), [SLOT_CURRENT] * N_PLAYERS, 'train', None))
            pool.submit(tasks)
        result = pool.get_result(timeout=1.)
        if result is None:
            continue
        if result['mode'] == 'eval':
            if eval_job is None or result['tag'] != eval_job['round']:
                continue
            eval_job['results'].append(result)
            if not result.get('failed'):
                _log_game('eval', result, len(eval_job['results']), eval_job['num_games'], eval_job['begin'])
            if len(eval_job['results']) == eval_job['num_games']:
                finish_eval(eval_job)
                eval_job = None
            continue
        if result.get('failed'):
            num_failed += 1
            logging.error(f'loop {n_loop}: self-play game {result["game_id"]} failed ({num_failed} so far)')
            continue
        iteration_results.append(result)
        _log_game('train', result, len(iteration_results), games_per_iteration, iteration_begin)
        if len(iteration_results) < games_per_iteration:
            continue

        # ======== iteration: the last games_per_iteration finished games
        iteration = state['iteration']
        results, iteration_results = iteration_results, list()
        self_play_seconds = time.time() - iteration_begin
        iteration_begin = time.time()
        reasons, avg_turns = summarize_games(results)
        num_decisions = sum(r['decisions'] for r in results)
        num_full = sum(r['full_searches'] for r in results)
        num_passes = sum(r.get('passes', 0) for r in results)
        logging.info(f'loop {n_loop} iteration {iteration}: {len(results)} self-play games in {self_play_seconds:.0f}s '
                     f'({len(results) / max(self_play_seconds, 1e-9):.2f} games/s), end_reasons={reasons}, avg_turns={avg_turns:.1f}, '
                     f'passes={num_passes}, full searches {num_full}/{num_decisions} decisions, '
                     f'{pool.outstanding["train"]} self-play games in progress')

        # -------- held-out check: the model has never trained on these games. A value loss far below the
        # held-out one on replay-buffer samples (already trained on) means memorisation
        recorded = [r['samples'] for r in results if len(r['samples']['obs'])]
        if recorded:
            held_out = model.evaluate_samples(*(np.concatenate([x[k] for x in recorded])
                                                for k in ('obs', 'probs', 'mask', 'values', 'value_weight')))
            message = f'loop {n_loop} iteration {iteration}: held-out (new games, before training): {_format_metrics(held_out)}'
            if model.memory_size() >= 1024:
                seen = model.evaluate_samples(*model.sample_memory(min(4096, model.memory_size())))
                message += f' | replay buffer (trained on): {_format_metrics(seen)}'
            logging.info(message)

        # -------- at most num_samples_per_game positions of every game go to the replay buffer / data file;
        # which ones is decided by the game's own seed
        num_new_data = 0
        for r in results:
            samples = r['samples']
            n = len(samples['obs'])
            rng = np.random.default_rng(derive_seed(r['seed'], SEED_SAMPLES))
            for i in np.sort(rng.choice(n, size=min(samples_per_game, n), replace=False)):
                sample = (samples['obs'][i], samples['probs'][i], samples['mask'][i], samples['values'][i],
                          float(samples['value_weight'][i]))
                if writer.write(*sample):
                    model.store_transition(*sample)
                    num_new_data += 1
        state['data_file_bytes'] = writer.flush()
        state['num_data'] += num_new_data
        state['num_data_written'] += num_new_data

        # -------- train: every sample is drawn train_sample_reuse times on average
        if state['num_data'] >= params['first_train_data_step']:
            state['train_credit'] += num_new_data * sample_reuse / model.batch_size
            num_train_steps = int(state['train_credit'])
            state['train_credit'] -= num_train_steps
            losses = list()
            for _ in range(num_train_steps):
                losses.append(model.learn())
                state['train_step_num'] += 1
            if losses:
                policy_loss, value_loss = np.mean(losses, axis=0)
                logging.info(f'loop {n_loop} iteration {iteration}: {num_new_data} new data (replay buffer {model.memory_size()}), '
                             f'{len(losses)} train steps (total {state["train_step_num"]}), action_probs_loss={policy_loss:.5f}, '
                             f'reward_value_loss={value_loss:.5f}, lr={model.scheduler.get_last_lr()[0]:.2e}')
            # -------- self-play continues with the new weights
            pool.set_weights(SLOT_CURRENT, get_state_dict_from_model(model), version=iteration + 1)
        else:
            logging.info(f'{state["num_data"]} data < first_train_data_step={params["first_train_data_step"]}, skip training.')

        state['iteration'] = iteration + 1
        # -------- eval new vs best (in the pool, alongside self-play)
        if (iteration + 1) % params['eval_model_per_iteration'] == 0:
            if eval_job is None:
                state['eval_round'] += 1
                new_state_dict = get_state_dict_from_model(model)
                save_weights(params, paths['eval'], new_state_dict, 'eval', iteration=iteration, eval_round=state['eval_round'])
                pool.set_weights(SLOT_NEW, new_state_dict, version=state['eval_round'])
                tasks = eval_tasks(params, state['eval_round'])
                pool.submit(tasks)
                eval_job = {'round': state['eval_round'], 'iteration': iteration, 'state_dict': new_state_dict,
                            'num_games': len(tasks), 'results': list(), 'begin': time.time()}
                logging.info(f'loop {n_loop} iteration {iteration}: evaluation {state["eval_round"]} started ({len(tasks)} games)')
            else:
                logging.info(f'loop {n_loop} iteration {iteration}: evaluation {eval_job["round"]} still running, no new one')
        save_training_state(params, paths, model, state)
        archive_every = params.get('checkpoint_archive_every', 0)
        if archive_every and state['iteration'] % archive_every == 0:
            save_weights(params, paths['archive'] % state['iteration'], get_state_dict_from_model(model), 'weights',
                         iteration=iteration)

    # -------- num_iterations reached: let a running evaluation finish
    while eval_job is not None and not interrupt.interrupt_callback() and _parent_alive():
        result = pool.get_result(timeout=1.)
        if result is not None and result['mode'] == 'eval' and result['tag'] == eval_job['round']:
            eval_job['results'].append(result)
            if len(eval_job['results']) == eval_job['num_games']:
                finish_eval(eval_job)
                eval_job = None
    logging.info(f'Finish loop {n_loop}: {state["iteration"]} iterations, train steps {state["train_step_num"]}, '
                 f'data {state["num_data"]}, {time.time() - run_begin:.0f}s.')


# ============================================================================ main
def validate_params(new_params, params_buffer):
    for historical_params in params_buffer:
        if historical_params == new_params:
            return False
    return True


SMOKE_PARAMS = {
    'num_iterations': 2,
    'num_self_play_games_per_iteration': 2,
    'eval_model_per_iteration': 1,
    'eval_task_num_games': 6,
    'max_game_turns': 12,
    'first_train_data_step': 1,
    'num_mcts_simulation_per_step': 8,
    'mcts_max_num_considered_actions': 8,
    'mcts_fast_num_simulation': 4,
    'mcts_fast_max_num_considered_actions': 4,
    'mcts_full_search_prob': 0.5,
    'eval_num_mcts_simulation_per_step': 8,
    'train_sample_reuse': 4.,
    'experiment_root': 'experiments_smoke',
    'init_weights_path': None,
    'warm_start_data_path': None,
    'eval_min_completion_rate': 0.,
    'model_param_dict': {'device': 'cpu', 'embedding_dim': 32, 'num_layers': 1, 'dim_feedforward': 64,
                         'transformer_head_dim': 16, 'batch_size': 8, 'num_inference_per_step': 1,
                         'inference_cuda_graph_batch_sizes': None},
}


def parse_args():
    parser = argparse.ArgumentParser(description='Train an AlphaGoZero agent for Splendor.')
    parser.add_argument('--mode', choices=['search', 'single'], default='search',
                        help='search: sample parameter combinations from --search-config (like test_train.py); single: one run with --default-config')
    parser.add_argument('--default-config', default='config/splendor_default.yml')
    parser.add_argument('--search-config', default='config/train_splendor.yml')
    parser.add_argument('--num-game-process', type=int, default=None, help='override num_game_process (0 = play in the train process)')
    parser.add_argument('--smoke', action='store_true', help='tiny CPU run to check the whole pipeline')
    parser.add_argument('--resume', metavar='EXPERIMENT_DIR', default=None,
                        help='continue an experiment exactly from its checkpoints/latest.pth, with its own config.yml '
                             '(strict: versions, architecture and every tensor must match)')
    parser.add_argument('--experiment-name', default=None, help='override experiment_name of the config')
    return parser.parse_args()


def experiment_dir_of(params, n_loop, search_mode):
    name = params.get('experiment_name') or time.strftime('splendor_%Y%m%d_%H%M%S')
    if search_mode:
        name = f'{name}_loop{n_loop}'
    return os.path.join(params.get('experiment_root', 'experiments'), name)


if __name__ == '__main__':
    args = parse_args()
    torch.multiprocessing.set_start_method('spawn')
    if args.resume:
        experiment_dir = os.path.normpath(args.resume)
        resume_params = yaml.load(open(os.path.join(experiment_dir, 'config.yml'), encoding='UTF-8'), Loader=yaml.FullLoader)
        resume_params['experiment_dir'] = experiment_dir
        if args.num_game_process is not None:
            resume_params['num_game_process'] = args.num_game_process
        log_level = getattr(logging, str(resume_params.get('log_level', 'INFO')).upper())
        init_logging(log_level)
        p = mp.Process(target=train_process, args=(resume_params, 0, log_level, True))
        p.start()
        p.join()
        raise SystemExit(p.exitcode)

    default_params = yaml.load(open(args.default_config, encoding='UTF-8'), Loader=yaml.FullLoader)
    log_level = getattr(logging, str(default_params.get('log_level', 'INFO')).upper())
    init_logging(log_level)

    if args.smoke:
        default_params = update_concurrent(copy.deepcopy(default_params), copy.deepcopy(SMOKE_PARAMS))
        default_params['experiment_name'] = time.strftime('smoke_%Y%m%d_%H%M%S')
        args.mode = 'single'
    if args.num_game_process is not None:
        default_params['num_game_process'] = args.num_game_process
    if args.experiment_name is not None:
        default_params['experiment_name'] = args.experiment_name

    if args.mode == 'single':
        task_params_list = [dict()]
    else:
        search_params = yaml.load(open(args.search_config, encoding='UTF-8'), Loader=yaml.FullLoader)
        num_choices = search_params.pop('num_choices')
        task_params_list = list()
        for _ in range(num_choices * 20):
            if len(task_params_list) >= num_choices:
                break
            chosen_task_params = choose_params_concurrent(search_params)
            if validate_params(chosen_task_params, task_params_list):
                task_params_list.append(chosen_task_params)
        if len(task_params_list) < num_choices:
            logging.warning(f'only {len(task_params_list)} distinct parameter combinations available (num_choices={num_choices})')

    for i, chosen_task_params in enumerate(task_params_list):
        if interrupt.interrupt_callback():
            break
        logging.info(f'start train and eval loop {i}, chosen params: {chosen_task_params}')
        params = update_concurrent(copy.deepcopy(default_params), copy.deepcopy(chosen_task_params))
        params['experiment_dir'] = experiment_dir_of(params, i, args.mode == 'search')
        p = mp.Process(target=train_process, args=(params, i, log_level))
        p.start()
        p.join()
        logging.info(f'end train and eval loop {i}')
