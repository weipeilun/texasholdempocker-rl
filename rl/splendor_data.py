"""Training data file, version checks and seed derivation for the Splendor pipeline.

Data file (format 2)
    line 1: "#splendor-train-data " + JSON {"format": 2, rules / observation / action versions and layout hashes,
            discard_over_limit} (splendor.rl_env.version_info)
    then one sample per line, five tab-separated fields:
        observation   N_OBSERVATION integers (comma separated)
        policy        N_ACTIONS probabilities (comma separated, "0" for zero)
        legal mask    N_ACTIONS characters 0/1
        value         N_PLAYERS win shares relative to the acting player (comma separated)
        value weight  1, or 0 for a position of a truncated game (no result: no value supervision)
    A file whose header differs from the running code (other rules / encoding / action space) is refused as a
    whole; every sample is validated when it is written and when it is read (check_sample), invalid ones are
    dropped and counted.

Seeds
    Every random choice is derived from ids, not from which process plays what (derive_seed): an experiment's
    seed and a game's id give the game seed; a game seed and a decision index give that decision's search seed.
"""

import json
import logging
import os
from collections import deque

import numpy as np

from splendor.rl_env import FIELD_GROUPS, N_ACTIONS, N_OBSERVATION, N_PLAYERS, VOCAB_SIZES, version_info

DATA_FORMAT_VERSION = 2
CHECKPOINT_FORMAT_VERSION = 2
DATA_HEADER_PREFIX = '#splendor-train-data '

_FIELD_VOCAB = np.asarray([VOCAB_SIZES[g] for g in FIELD_GROUPS], dtype=np.int64)

# ---------------------------------------------------------------------------- seeds
# streams of an experiment seed s / eval seed e
SEED_TRAIN_GAME = 1     # derive_seed(s, SEED_TRAIN_GAME, game_id): seed of self-play game game_id
SEED_EVAL_GAME = 2      # derive_seed(e, SEED_EVAL_GAME, deal): seed of an evaluation deal (two games share one)
SEED_TRAINING = 3       # derive_seed(s, SEED_TRAINING): torch / numpy RNG of the train process
# streams of a game seed g
SEED_DEAL = 11          # the game's deal and real card draws
SEED_PLAYOUT_CAP = 12   # full / fast search choice of every decision
SEED_SEARCH = 13        # derive_seed(g, SEED_SEARCH, decision): that decision's search (Gumbel noise, chance draws)
SEED_SAMPLES = 14       # which recorded positions of the game become training samples
SEED_SYMMETRY = 15      # random symmetries of the network evaluations (mcts_random_symmetry)


def derive_seed(*ids):
    """64-bit seed of a tuple of non-negative integer ids (numpy SeedSequence: well mixed, independent streams)"""
    state = np.random.SeedSequence([int(i) for i in ids]).generate_state(2, dtype=np.uint32)
    return (int(state[0]) << 32) | int(state[1])


# ---------------------------------------------------------------------------- versions
def versions_mismatch(found, expected):
    """-> list of 'key: found != expected' for every differing entry (empty when they agree)"""
    found = found or dict()
    return [f'{k}: {found.get(k)!r} != {expected.get(k)!r}' for k in sorted(set(found) | set(expected))
            if found.get(k) != expected.get(k)]


# ---------------------------------------------------------------------------- samples
def check_sample(obs, probs, mask, values, value_weight):
    """-> None if the sample is valid, else the reason"""
    obs, probs, mask, values = np.asarray(obs), np.asarray(probs), np.asarray(mask), np.asarray(values)
    if obs.shape != (N_OBSERVATION,) or probs.shape != (N_ACTIONS,) or mask.shape != (N_ACTIONS,) or values.shape != (N_PLAYERS,):
        return f'shapes {obs.shape} {probs.shape} {mask.shape} {values.shape}'
    if not np.issubdtype(obs.dtype, np.integer) or obs.min() < 0 or (obs >= _FIELD_VOCAB).any():
        bad = np.flatnonzero((obs < 0) | (obs >= _FIELD_VOCAB))
        return f'observation out of its vocab at fields {bad[:5].tolist()}'
    mask = mask.astype(bool)
    if not mask.any():
        return 'no legal action'
    if not np.isfinite(probs).all() or probs.min() < -1e-7 or abs(float(probs.sum()) - 1.) > 1e-3:
        return f'policy is not a distribution (sum {float(probs.sum()):.6f})'
    if (~mask).any() and float(probs[~mask].max()) > 1e-6:
        return 'policy puts mass on illegal actions'
    if not np.isfinite(values).all() or values.min() < 0. or values.max() > 1. + 1e-6:
        return f'value outside [0, 1]: {values.tolist()}'
    if value_weight == 1:
        if abs(float(values.sum()) - 1.) > 1e-4:
            return f'value of a finished game does not sum to 1: {values.tolist()}'
    elif value_weight == 0:
        if values.any():
            return 'value of a truncated game (weight 0) is not zero'
    else:
        return f'value weight {value_weight!r} is not 0 / 1'
    return None


def _format_prob(p):
    return '0' if p == 0 else ('%.5f' % p).rstrip('0').rstrip('.')


def format_sample(obs, probs, mask, values, value_weight):
    return '\t'.join((','.join(str(int(v)) for v in obs),
                      ','.join(_format_prob(float(p)) for p in probs),
                      ''.join('1' if m else '0' for m in mask),
                      ','.join('%.6f' % float(v) for v in values),
                      '1' if value_weight else '0')) + '\n'


def parse_sample(line):
    obs, probs, mask, values, weight = line.rstrip('\n').split('\t')
    return (np.asarray([int(v) for v in obs.split(',')], dtype=np.int32),
            np.asarray([float(v) for v in probs.split(',')], dtype=np.float32),
            np.frombuffer(mask.encode('ascii'), dtype=np.uint8) == ord('1'),
            np.asarray([float(v) for v in values.split(',')], dtype=np.float32),
            float(int(weight)))


# ---------------------------------------------------------------------------- data file
def read_header(path):
    """-> the header dict of a data file, or None for an empty / missing file; ValueError for a file without
    a format-2 header (written by an older pipeline)"""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    with open(path, 'r', encoding='UTF-8') as f:
        first = f.readline()
    if not first.startswith(DATA_HEADER_PREFIX):
        raise ValueError(f'{path} has no "{DATA_HEADER_PREFIX.strip()}" header: written by an older pipeline '
                         f'(unknown rules / encoding version), not usable')
    return json.loads(first[len(DATA_HEADER_PREFIX):])


def expected_header(discard_over_limit=True):
    return dict(version_info(discard_over_limit), format=DATA_FORMAT_VERSION)


def check_data_file(path, discard_over_limit=True):
    """raise ValueError unless the file's header matches the running code"""
    header = read_header(path)
    if header is None:
        return
    mismatch = versions_mismatch(header, expected_header(discard_over_limit))
    if mismatch:
        raise ValueError(f'{path} was written for other rules / encoding / format: {"; ".join(mismatch)}')


class TrainDataWriter:
    """Appends validated samples to a data file (writes the header to a new file, checks it on an existing one)."""

    def __init__(self, path, discard_over_limit=True):
        self.path = path
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        check_data_file(path, discard_over_limit)
        new = read_header(path) is None
        self.file = open(path, 'a', encoding='UTF-8')
        if new:
            self.file.write(DATA_HEADER_PREFIX + json.dumps(expected_header(discard_over_limit), sort_keys=True) + '\n')
            self.file.flush()
        self.num_rejected = 0

    def write(self, obs, probs, mask, values, value_weight):
        """-> True if written; an invalid sample is logged and dropped"""
        reason = check_sample(obs, probs, mask, values, value_weight)
        if reason is not None:
            self.num_rejected += 1
            logging.error(f'invalid training sample not written ({reason})')
            return False
        self.file.write(format_sample(obs, probs, mask, values, value_weight))
        return True

    def flush(self):
        """flush and return the file size (a consistent point to resume from)"""
        self.file.flush()
        os.fsync(self.file.fileno())
        return os.path.getsize(self.path)

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None


def read_samples(path, discard_over_limit=True, max_bytes=None, last=None):
    """Validated samples of a data file -> (list of (obs, probs, mask, values, value_weight), stats dict).

    max_bytes: read only the first max_bytes bytes (a resume point); last: keep only the last `last` samples.
    Raises ValueError if the header does not match the running code."""
    check_data_file(path, discard_over_limit)
    keep = deque(maxlen=last) if last else list()
    stats = {'read': 0, 'invalid': 0, 'reasons': dict()}
    if read_header(path) is None:
        return list(keep), stats
    with open(path, 'rb') as f:
        position = len(f.readline())   # header
        while True:
            if max_bytes is not None and position >= max_bytes:
                break
            line = f.readline()
            if not line:
                break
            position += len(line)
            if not line.endswith(b'\n'):
                break   # a partly written last line (killed while writing)
            try:
                sample = parse_sample(line.decode('ascii'))
                reason = check_sample(*sample)
            except (ValueError, IndexError) as e:
                reason = f'unparsable ({type(e).__name__})'
            if reason is not None:
                stats['invalid'] += 1
                key = reason.split(' (')[0].split(':')[0]
                stats['reasons'][key] = stats['reasons'].get(key, 0) + 1
                continue
            stats['read'] += 1
            keep.append(sample)
    if stats['invalid']:
        logging.warning(f'{path}: dropped {stats["invalid"]} invalid samples: {stats["reasons"]}')
    return list(keep), stats


def truncate_data_file(path, size):
    """cut the file back to `size` bytes (samples written after the checkpoint being resumed)"""
    if os.path.exists(path) and os.path.getsize(path) > size:
        with open(path, 'r+b') as f:
            f.truncate(size)
        return True
    return False
