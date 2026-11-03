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
    best one in evaluation games. Predictions are batched in predict processes, game
    loops run as threads inside worker processes, as in predict_batch_process /
    train_eval_process.

Splendor has no external train data, so every iteration of train_process is
    self-play -> train -> (every N iterations) eval new vs best.

Usage (from the repository root):
    python train_splendor.py                  # parameter search, config/train_splendor.yml
    python train_splendor.py --mode single    # one run with config/splendor_default.yml only
    python train_splendor.py --mode single --smoke   # tiny CPU run to check the pipeline
"""

import argparse
import copy
import logging
import math
import os
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
from rl.splendor_mcts import SplendorMCTS
from rl.splendor_model import SplendorAlphaGoZero, save_model, load_model, splendor_train_data_generator, resolve_device
from splendor.rl_env import SplendorRLEnv, to_relative, N_ACTIONS, ACTION_NAMES

LOG_FORMAT = '%(asctime)s.%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d,%H:%M:%S'

# 评估时新/旧模型各占两个座位。相邻两局使用同一种子、互补的座位安排，抵消座位与发牌的影响。
EVAL_SEAT_ARRANGEMENTS = [(1, 0, 1, 0), (0, 1, 0, 1), (1, 1, 0, 0), (0, 0, 1, 1), (1, 0, 0, 1), (0, 1, 1, 0)]
BEST_MODEL_IDX, NEW_MODEL_IDX = 0, 1


def init_logging(log_level):
    logging.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    logging.getLogger().setLevel(log_level)


# ============================================================================ game
def play_game(game_id, seed, predictor_by_seat, params, mode, stop_event=None):
    """Play one 4-player game with MCTS at every decision.

    Gumbel AlphaZero (rl/splendor_mcts.py): the played move is the action selected by Sequential
    Halving with Gumbel, the policy target is the improved policy pi' = softmax(logits + sigma(completed_q)).
    mode 'train': gumbel_scale = mcts_gumbel_scale (sampled moves), returns training samples.
    mode 'eval':  gumbel_scale = 0 (greedy).
    Returns None when interrupted.
    """
    is_train = mode == 'train'
    env = SplendorRLEnv(max_turns=params['max_game_turns'])
    env.reset(seed)

    def should_stop():
        return interrupt.interrupt_callback() or (stop_event is not None and stop_event.is_set())

    n_simulation = params['num_mcts_simulation_per_step'] if is_train else params['eval_num_mcts_simulation_per_step']
    trajectory = list()
    t0 = time.time()
    while not env.terminated:
        seat = env.acting_player
        mcts = SplendorMCTS(N_ACTIONS,
                            predictor=predictor_by_seat[seat],
                            n_simulation=n_simulation,
                            max_num_considered_actions=params['mcts_max_num_considered_actions'],
                            gumbel_scale=params['mcts_gumbel_scale'] if is_train else 0.,
                            value_scale=params['mcts_value_scale'],
                            maxvisit_init=params['mcts_maxvisit_init'],
                            rescale_values=params['mcts_rescale_values'],
                            rollout_lambda=params['mcts_rollout_lambda'],
                            rollout_policy=params['mcts_rollout_policy'],
                            max_search_depth=params['mcts_max_search_depth'],
                            interrupt_callback=should_stop)
        observation = env.observation()
        action_mask = env.legal_mask().astype(np.int32)
        search_result = mcts.simulate(env)
        if search_result is None:
            return None
        action_bin, policy_target = search_result
        if is_train:
            trajectory.append((observation, policy_target.astype(np.float32), action_mask, seat))
        env.step(action_bin)

    rewards = env.final_rewards()
    result = {
        'game_id': game_id,
        'seed': seed,
        'rewards': rewards,
        'end_reason': env.end_reason,
        'turns': env.game.turn,
        'points': [p.points for p in env.game.players],
        'seconds': time.time() - t0,
    }
    if is_train:
        # 终局收益按“相对当前行动玩家”的座位顺序存储，与模型 value 头的输出顺序一致
        result['samples'] = [(obs, probs, mask, to_relative(rewards, seat).astype(np.float32)) for obs, probs, mask, seat in trajectory]
    return result


# ============================================================================ predict process
def predict_batch_process(server_id, model_param_dict, state_dict, in_queue, result_conn_table, predict_batch_size, predict_batch_wait_seconds, stop_event, log_level):
    """Batched inference for one model version (best or new), PyTorch only.

    result_conn_table[worker_id][thread_id]: write end of that game thread's own result pipe.
    Each game thread waits for one prediction at a time, so a pipe never holds more than one
    small message and Connection.send never blocks.
    """
    init_logging(log_level)
    model_param_dict = dict(model_param_dict, save_train_data=False, transition_buffer_len=1)
    model = SplendorAlphaGoZero(**model_param_dict)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    logging.info(f'predict_batch_process {server_id} pid {os.getpid()} started on {model.device}')

    num_predicted = 0
    last_log_time = time.time()
    while not stop_event.is_set() and not interrupt.interrupt_callback():
        try:
            first = in_queue.get(block=True, timeout=0.1)
        except Empty:
            continue
        batch = [first]
        deadline = time.time() + predict_batch_wait_seconds
        while len(batch) < predict_batch_size:
            try:
                batch.append(in_queue.get(block=True, timeout=max(0., deadline - time.time())))
            except Empty:
                break

        worker_ids, thread_ids, observations, masks = zip(*batch)
        probs, values = model.predict_batch(observations, masks)
        for worker_id, thread_id, p, v in zip(worker_ids, thread_ids, probs, values):
            result_conn_table[worker_id][thread_id].send((p, v))

        num_predicted += len(batch)
        now = time.time()
        if now - last_log_time > 120:
            logging.info(f'predict_batch_process {server_id}: {num_predicted / (now - last_log_time):.1f} predictions/s, in_queue qsize:{_qsize(in_queue)}')
            num_predicted = 0
            last_log_time = now
    logging.info(f'predict_batch_process {server_id} exit')


def _qsize(q):
    try:
        return q.qsize()
    except NotImplementedError:  # macOS
        return -1


# ============================================================================ game worker process
def game_worker_process(worker_id, params, mode, task_queue, result_queue, predict_in_queue_list, result_conn_list, stop_event, log_level):
    """Runs one game loop thread per result pipe; predictions go to the predict processes (as train_eval_process).

    Every thread reads its predictions from its own pipe (result_conn_list[thread_id]), so no
    dispatcher thread is needed and a result costs no extra GIL hand-off inside this process.
    """
    init_logging(log_level)
    num_threads = len(result_conn_list)

    def should_stop():
        return stop_event.is_set() or interrupt.interrupt_callback()

    def make_predictor(thread_id, server_id):
        result_conn = result_conn_list[thread_id]

        def predictor(observation, mask):
            predict_in_queue_list[server_id].put((worker_id, thread_id, observation, mask))
            begin_time = time.time()
            # poll() releases the GIL while waiting
            while not result_conn.poll(0.1):
                if should_stop():
                    # 返回均匀分布，MCTS 会在下一次检查中断时退出，本线程不再发请求
                    return mask / mask.sum(), np.zeros(4, dtype=np.float32)
                if time.time() - begin_time > 60:
                    logging.warning(f'game_worker {worker_id} thread {thread_id} waited prediction for {time.time() - begin_time:.0f}s')
                    begin_time = time.time()
            return result_conn.recv()
        return predictor

    def game_loop_thread(thread_id):
        while not should_stop():
            try:
                task = task_queue.get(block=True, timeout=0.5)
            except Empty:
                continue
            if task is None:
                break
            game_id, seed, seat_server_ids = task
            predictor_by_seat = [make_predictor(thread_id, server_id) for server_id in seat_server_ids]
            result = play_game(game_id, seed, predictor_by_seat, params, mode, stop_event)
            if result is None:
                break
            result['seat_server_ids'] = seat_server_ids
            result_queue.put(result)

    threads = [threading.Thread(target=game_loop_thread, args=(tid,), daemon=True) for tid in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        while t.is_alive() and not should_stop():
            t.join(timeout=0.5)
    logging.info(f'game_worker_process {worker_id} exit')


# ============================================================================ run a batch of games
def run_games(params, state_dict_list, tasks, mode, log_level):
    """Play `tasks` = [(game_id, seed, seat_server_ids)], one predict server per state dict.

    num_game_process == 0 plays everything in this process (debug / no GPU).
    """
    model_param_dict = params['model_param_dict']
    results = list()
    begin_time = time.time()

    if params['num_game_process'] <= 0:
        models = list()
        for state_dict in state_dict_list:
            m = SplendorAlphaGoZero(**dict(model_param_dict, save_train_data=False, transition_buffer_len=1))
            m.load_state_dict(state_dict, strict=False)
            models.append(m)
        for game_id, seed, seat_server_ids in tasks:
            if interrupt.interrupt_callback():
                break
            result = play_game(game_id, seed, [models[s].predict for s in seat_server_ids], params, mode)
            if result is None:
                break
            result['seat_server_ids'] = seat_server_ids
            results.append(result)
            _log_game(mode, result, len(results), len(tasks), begin_time)
        return results

    ctx = mp.get_context('spawn')
    num_game_process = params['num_game_process']
    num_threads = params['num_game_thread_per_process']
    stop_event = ctx.Event()
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    predict_in_queue_list = [ctx.Queue() for _ in state_dict_list]
    # 每个对局线程一条专属的单向回传 pipe：推理进程持有写端，对局线程持有读端
    _ensure_fd_limit(4 * num_game_process * num_threads + 256)
    pipes = [[ctx.Pipe(duplex=False) for _ in range(num_threads)] for _ in range(num_game_process)]
    result_recv_conn_table = [[recv_conn for recv_conn, _ in row] for row in pipes]
    result_send_conn_table = [[send_conn for _, send_conn in row] for row in pipes]

    processes = list()
    for server_id, state_dict in enumerate(state_dict_list):
        p = ctx.Process(target=predict_batch_process, args=(server_id, model_param_dict, state_dict, predict_in_queue_list[server_id], result_send_conn_table, params['predict_batch_size'], params['predict_batch_wait_seconds'], stop_event, log_level), daemon=True)
        p.start()
        processes.append(p)
    for worker_id in range(num_game_process):
        p = ctx.Process(target=game_worker_process, args=(worker_id, params, mode, task_queue, result_queue, predict_in_queue_list, result_recv_conn_table[worker_id], stop_event, log_level), daemon=True)
        p.start()
        processes.append(p)

    for task in tasks:
        task_queue.put(task)
    for _ in range(num_game_process * num_threads):
        task_queue.put(None)

    while len(results) < len(tasks):
        if interrupt.interrupt_callback():
            logging.info('run_games detect interrupt')
            break
        try:
            result = result_queue.get(block=True, timeout=1.)
        except Empty:
            if not any(p.is_alive() for p in processes[len(state_dict_list):]):
                logging.error('all game workers exited before finishing the tasks')
                break
            continue
        results.append(result)
        _log_game(mode, result, len(results), len(tasks), begin_time)

    stop_event.set()
    for p in processes:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()
    # 每次 run_games 都新建 pipe，用完关闭，避免多个 iteration 累积文件描述符
    for row in pipes:
        for recv_conn, send_conn in row:
            recv_conn.close()
            send_conn.close()
    return results


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
                 f'turns={result["turns"]}, points={result["points"]}, seats={result["seat_server_ids"]}, '
                 f'game cost {result["seconds"]:.1f}s, total {time.time() - begin_time:.1f}s')


def summarize_games(results):
    reasons = dict()
    for r in results:
        reasons[r['end_reason']] = reasons.get(r['end_reason'], 0) + 1
    avg_turns = float(np.mean([r['turns'] for r in results])) if results else 0.
    return reasons, avg_turns


# ============================================================================ train
def train_on_historical_data(model, params):
    path = params['train_data_path']
    if not params['train_with_historical_data'] or not os.path.exists(path):
        logging.info('No historical train data used, start training from self-play.')
        return 0
    logging.info(f'Found historical train data from {path}. Train with this first.')
    num_data, train_step_num = 0, 0
    next_train_step = params['first_train_data_step']
    for observation, action_probs, action_mask, reward_values in splendor_train_data_generator(path):
        if interrupt.interrupt_callback():
            break
        model.store_transition(observation, action_probs, action_mask, reward_values, save_train_data=False)
        num_data += 1
        if num_data >= next_train_step:
            action_probs_loss, reward_value_loss = model.learn()
            train_step_num += 1
            if train_step_num % params['log_step_num'] == 0:
                logging.info(f'historical train_step {train_step_num}, action_probs_loss={action_probs_loss:.5f}, reward_value_loss={reward_value_loss:.5f}')
            # 避免训练数据过多重复
            next_train_step += params['train_per_step']
    logging.info(f'Finished training historical data: {num_data} data, {train_step_num} train steps.')
    return num_data


def evaluate(params, best_state_dict, new_state_dict, n_loop, iteration, log_level):
    num_games = params['eval_task_num_games']
    base_seed = params['eval_seed']
    tasks = list()
    for k in range(num_games):
        arrangement = EVAL_SEAT_ARRANGEMENTS[k % len(EVAL_SEAT_ARRANGEMENTS)]
        tasks.append((k, base_seed + k // 2, list(arrangement)))
    results = run_games(params, [best_state_dict, new_state_dict], tasks, 'eval', log_level)

    new_reward_sum, num_decided = 0., 0
    for r in results:
        if r['end_reason'] != 'score':
            continue
        num_decided += 1
        new_reward_sum += sum(reward for reward, server_id in zip(r['rewards'], r['seat_server_ids']) if server_id == NEW_MODEL_IDX)
    # 新模型占一半座位，势均力敌时胜率为 0.5
    new_model_win_rate = new_reward_sum / num_decided if num_decided > 0 else 0.
    reasons, avg_turns = summarize_games(results)
    logging.info(f'Evaluation finished for loop={n_loop}, iteration={iteration}: new_model_win_rate={new_model_win_rate:.4f} '
                 f'over {num_decided} decided games, end_reasons={reasons}, avg_turns={avg_turns:.1f}')
    return new_model_win_rate, len(results) == num_games


def train_process(params, n_loop, log_level):
    init_logging(log_level)
    torch.manual_seed(params['seed'] + n_loop)
    np.random.seed(params['seed'] + n_loop)

    for path_key in ('train_data_path', 'model_save_checkpoint_path_format'):
        directory = os.path.dirname(params[path_key])
        if directory:
            os.makedirs(directory, exist_ok=True)

    model_param_dict = params['model_param_dict']
    model_param_dict['device'] = resolve_device(model_param_dict.get('device'))
    model_param_dict['default_train_file_path'] = params['train_data_path']
    logging.info(f'loop {n_loop} params: {params}')

    # 先读历史数据（模型初始化时会以追加模式打开同一个文件）
    model = SplendorAlphaGoZero(**dict(model_param_dict, save_train_data=False))
    init_path = params.get('model_init_checkpoint_path')
    if init_path and os.path.exists(init_path):
        load_model(model, init_path)
    num_historical_data = train_on_historical_data(model, params)
    if model_param_dict.get('save_train_data', False):
        model.train_file = open(params['train_data_path'], 'a', encoding='UTF-8')

    model_save_checkpoint_path = params['model_save_checkpoint_path_format'] % n_loop
    model_best_checkpoint_path = params['model_best_checkpoint_path_format'] % n_loop
    model_eval_snapshot_path = params['model_eval_snapshot_path_format'] % n_loop
    save_model(model, model_save_checkpoint_path)

    best_state_dict = get_state_dict_from_model(model)
    save_model(model, model_best_checkpoint_path)

    seed = params['seed'] * 1000003 + n_loop * 100003
    game_id = 0
    num_data = num_historical_data
    train_step_num = 0
    for iteration in range(params['num_iterations']):
        if interrupt.interrupt_callback():
            break
        # -------- self-play
        current_state_dict = get_state_dict_from_model(model)
        tasks = list()
        for _ in range(params['num_self_play_games_per_iteration']):
            game_id += 1
            seed += 1
            tasks.append((game_id, seed, [0, 0, 0, 0]))
        logging.info(f'loop {n_loop} iteration {iteration}: self-play {len(tasks)} games')
        results = run_games(params, [current_state_dict], tasks, 'train', log_level)
        if interrupt.interrupt_callback():
            break

        num_new_data = 0
        for r in results:
            if params['skip_unfinished_games'] and r['end_reason'] != 'score':
                continue
            for observation, action_probs, action_mask, reward_values in r['samples']:
                model.store_transition(observation, action_probs, action_mask, reward_values)
                num_new_data += 1
        model.flush_train_data()
        num_data += num_new_data
        reasons, avg_turns = summarize_games(results)
        logging.info(f'loop {n_loop} iteration {iteration}: {len(results)} self-play games, {num_new_data} new data, '
                     f'end_reasons={reasons}, avg_turns={avg_turns:.1f}')

        # -------- train
        if num_data < params['first_train_data_step']:
            logging.info(f'{num_data} data < first_train_data_step={params["first_train_data_step"]}, skip training.')
            continue
        num_train_steps = max(params['min_train_steps_per_iteration'], math.ceil(num_new_data / params['train_per_step']))
        for _ in range(num_train_steps):
            if interrupt.interrupt_callback():
                break
            try:
                action_probs_loss, reward_value_loss = model.learn()
            except ValueError as e:
                logging.error(f'ValueError in model.learn: {e}')
                model.close()
                return
            train_step_num += 1
            if train_step_num % params['log_step_num'] == 0:
                logging.info(f'train_step {train_step_num}, action_probs_loss={action_probs_loss:.5f}, reward_value_loss={reward_value_loss:.5f}, lr={model.scheduler.get_last_lr()[0]:.2e}')
        save_model(model, model_save_checkpoint_path)

        # -------- eval new vs best
        if (iteration + 1) % params['eval_model_per_iteration'] == 0:
            new_state_dict = get_state_dict_from_model(model)
            save_model(model, model_eval_snapshot_path)
            new_model_win_rate, finished = evaluate(params, best_state_dict, new_state_dict, n_loop, iteration, log_level)
            if finished and new_model_win_rate > params['update_model_win_rate_thres']:
                best_state_dict = new_state_dict
                save_model(model, model_best_checkpoint_path)
                logging.info(f'loop {n_loop} iteration {iteration}: new model promoted to best ({model_best_checkpoint_path})')
            else:
                logging.info(f'loop {n_loop} iteration {iteration}: best model kept')

    model.close()
    logging.info(f'Finish loop {n_loop}, train steps {train_step_num}, data {num_data}.')


# ============================================================================ main
def validate_params(new_params, params_buffer):
    for historical_params in params_buffer:
        if historical_params == new_params:
            return False
    return True


SMOKE_PARAMS = {
    'num_iterations': 2,
    'num_self_play_games_per_iteration': 1,
    'eval_model_per_iteration': 1,
    'eval_task_num_games': 2,
    'max_game_turns': 12,
    'first_train_data_step': 1,
    'min_train_steps_per_iteration': 2,
    'num_mcts_simulation_per_step': 8,
    'eval_num_mcts_simulation_per_step': 8,
    'train_with_historical_data': False,
    'train_data_path': 'data/splendor_smoke.txt',
    'model_init_checkpoint_path': None,
    'model_save_checkpoint_path_format': 'models/splendor_smoke_%d.pth',
    'model_best_checkpoint_path_format': 'models/splendor_smoke_%d_best.pth',
    'model_eval_snapshot_path_format': 'models/splendor_smoke_%d_eval.pth',
    'model_param_dict': {'device': 'cpu', 'embedding_dim': 32, 'num_layers': 1, 'dim_feedforward': 64,
                         'transformer_head_dim': 16, 'batch_size': 8, 'num_inference_per_step': 1},
}


def parse_args():
    parser = argparse.ArgumentParser(description='Train an AlphaGoZero agent for Splendor.')
    parser.add_argument('--mode', choices=['search', 'single'], default='search',
                        help='search: sample parameter combinations from --search-config (like test_train.py); single: one run with --default-config')
    parser.add_argument('--default-config', default='config/splendor_default.yml')
    parser.add_argument('--search-config', default='config/train_splendor.yml')
    parser.add_argument('--num-game-process', type=int, default=None, help='override num_game_process (0 = play in the train process)')
    parser.add_argument('--smoke', action='store_true', help='tiny CPU run to check the whole pipeline')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    default_params = yaml.load(open(args.default_config, encoding='UTF-8'), Loader=yaml.FullLoader)
    log_level = getattr(logging, str(default_params.get('log_level', 'INFO')).upper())
    init_logging(log_level)

    if args.smoke:
        default_params = update_concurrent(copy.deepcopy(default_params), copy.deepcopy(SMOKE_PARAMS))
        args.mode = 'single'
    if args.num_game_process is not None:
        default_params['num_game_process'] = args.num_game_process

    torch.multiprocessing.set_start_method('spawn')

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
        p = mp.Process(target=train_process, args=(params, i, log_level))
        p.start()
        p.join()
        logging.info(f'end train and eval loop {i}')
