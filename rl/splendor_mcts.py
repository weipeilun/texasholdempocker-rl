"""Gumbel AlphaZero MCTS for Splendor.

Search algorithm: "Policy improvement by planning with Gumbel" (Danihelka, Guez, Schrittwieser,
Silver, ICLR 2022), following DeepMind's reference implementation mctx
(gumbel_muzero_policy, gumbel_muzero_root_action_selection, gumbel_muzero_interior_action_selection,
qtransform_completed_by_mix_value, seq_halving). Differences to mctx are marked "[splendor]".

Root
    * g(a) ~ gumbel_scale * Gumbel(0, 1) for every action, sampled once per search.
    * Sequential Halving over the m = min(max_num_considered_actions, #legal) actions with the
      highest g(a) + logits(a): simulation i visits, among the actions whose visit count equals
      table[m][i] (seq_halving schedule), the one with the highest g + logits + sigma(q_hat).
      This samples the top-m actions without replacement (Gumbel-top-k), then repeatedly keeps
      the better half, without any explicit elimination step.
    * selected action (the move actually played): among the actions with the maximal visit
      count, argmax g + logits + sigma(q_hat). With gumbel_scale=0 this is a greedy choice.
    * policy target pi' = softmax(logits + sigma(completed_q)) over legal actions.
      No visit counts, no temperature: schedule-driven visits cannot leak into it.
Interior nodes
    * deterministic selection argmax_a pi'(a) - N(a) / (1 + sum_b N(b)), which makes the visit
      frequencies track pi' = softmax(logits + sigma(completed_q)).
      [splendor] illegal actions are masked (mctx's MuZero does not know legality).
Q transform (qtransform_completed_by_mix_value)
    * unvisited actions get the mixed value (v(s) + sum_N * prior-weighted Q of visited
      actions) / (sum_N + 1); completed Q is min-max rescaled to [0, 1] within the node
      ([splendor] over legal actions only) and multiplied by (maxvisit_init + max_a N(a)) * value_scale.

Tree / evaluation
    * closed loop: decision nodes are exact states with the prior, legal mask and v(s) stored at
      expansion (one model evaluation per simulation); card refills are explicit chance nodes,
      one draw sampled per simulation with its probability.
    * leaf value V(s_L) = (1 - λ) v_θ(s_L) + λ z_L (AlphaGo; λ = 0: value network only).
    * a game ended by the rules (score / draw) inside the tree backs up its true outcome. A state truncated at
      max_turns has no outcome: it is a leaf evaluated by the network (bootstrap), never "nobody wins";
      a rollout that hits max_turns uses v_θ for z.
    * 4 players: values are absolute-seat vectors; every node uses its own acting player's
      component (max^n), no alternating negation.

Implementation
    * the per-node numerics live in a Node object: rl/_mcts_core.c (C, build with
      `python setup_mcts_core.py build_ext --inplace`) or PyNode below (numpy reference). Both
      give bit-identical results; the search uses the C node when it is built.
    * predictor(request: bytes) -> reply: bytes, the wire format of train_splendor.py:
      request = observation (N_OBSERVATION uint8) + legal mask (N_ACTIONS uint8),
      reply = float32 [N_ACTIONS action probabilities + N_PLAYERS values relative to the acting player].
"""

import math
import random
from functools import lru_cache

import numpy as np

from splendor.bots import choose_action as bot_choose_action, evaluate_position, WIN_VALUE
from splendor.rl_env import to_absolute, action_to_index, N_PLAYERS

try:
    from rl._mcts_core import Node as CNode
except ImportError:   # not built: numpy reference implementation
    CNode = None

PREDICTOR_INTERFACE = 'bytes'
_LOW_LOGIT = -1e9   # mctx score_considered floor
_TINY = np.finfo(np.float64).tiny


# ---------------------------------------------------------------------------- Sequential Halving
# Verbatim port of mctx/_src/seq_halving.py (Apache License 2.0, Copyright 2021 DeepMind Technologies Limited).
def get_sequence_of_considered_visits(max_num_considered_actions, num_simulations):
    """Visit count that the action visited by simulation i must currently have."""
    if max_num_considered_actions <= 1:
        return tuple(range(num_simulations))
    log2max = int(math.ceil(math.log2(max_num_considered_actions)))
    sequence = []
    visits = [0] * max_num_considered_actions
    num_considered = max_num_considered_actions
    while len(sequence) < num_simulations:
        num_extra_visits = max(1, int(num_simulations / (log2max * num_considered)))
        for _ in range(num_extra_visits):
            sequence.extend(visits[:num_considered])
            for i in range(num_considered):
                visits[i] += 1
        # Halving the number of considered actions.
        num_considered = max(2, num_considered // 2)
    return tuple(sequence[:num_simulations])


@lru_cache(maxsize=None)
def get_table_of_considered_visits(max_num_considered_actions, num_simulations):
    return tuple(get_sequence_of_considered_visits(m, num_simulations)
                 for m in range(max_num_considered_actions + 1))


# ---------------------------------------------------------------------------- tree
class PyNode:
    """numpy reference of rl._mcts_core.Node (same interface, bit-identical results).

    Node(seat, reply, mask): reply = float32 bytes [n_actions probabilities + N_PLAYERS values relative
    to seat], mask = uint8 bytes [n_actions]. Children: action -> node, or for chance outcomes
    action -> dict (drawn card -> node).
    """
    __slots__ = ('seat', 'logits', 'prior', 'mask', 'value', 'children', 'w', 'n', 'q', 'num_legal', 'visits', 'gumbel')

    def __init__(self, seat, reply, mask, _clone_of=None):
        if _clone_of is not None:
            src = _clone_of
            self.seat, self.mask, self.prior, self.logits, self.value = src.seat, src.mask, src.prior, src.logits, src.value
            self.num_legal = src.num_legal
        else:
            n_actions = len(mask)
            mask = np.frombuffer(mask, dtype=np.uint8).astype(bool)
            reply = np.frombuffer(reply, dtype=np.float32)
            probs = np.asarray(reply[:n_actions], dtype=np.float64) * mask
            total = probs.sum()
            probs = probs / total if total > 0 else mask / mask.sum()
            self.seat = seat                 # acting player of this state
            self.mask = mask                 # legal actions
            self.prior = probs               # P(s, .), masked and normalised
            # prior logits, max-normalised over legal actions; illegal = -inf (only used under the mask)
            logits = np.where(mask, np.log(np.maximum(probs, _TINY)), -np.inf)
            self.logits = logits - logits[mask].max()
            # v_θ(s), absolute seats
            self.value = tuple(to_absolute(reply[n_actions:n_actions + N_PLAYERS], seat).astype(np.float64).tolist())
            self.num_legal = int(mask.sum())
        n_actions = len(self.mask)
        self.children = dict()
        self.w = np.zeros(n_actions, dtype=np.float64)
        self.n = np.zeros(n_actions, dtype=np.float64)
        self.q = np.zeros(n_actions, dtype=np.float64)   # from self.seat's perspective
        self.visits = 0
        self.gumbel = None

    def clone_fresh(self):
        return PyNode(None, None, None, _clone_of=self)

    def set_gumbel(self, gumbel):
        self.gumbel = np.asarray(gumbel, dtype=np.float64)

    def _completed_qvalues(self, maxvisit_init, value_scale, rescale_values):
        """sigma(completed Q) of qtransform_completed_by_mix_value; illegal entries are meaningless."""
        n, q, mask = self.n, self.q, self.mask
        visited = n > 0
        raw_value = self.value[self.seat]
        sum_n = n.sum()
        if sum_n > 0:
            prior = np.maximum(self.prior, _TINY)
            sum_probs = prior[visited].sum()
            weighted_q = (prior[visited] * q[visited]).sum() / sum_probs
            mixed_value = (raw_value + sum_n * weighted_q) / (sum_n + 1)
        else:
            mixed_value = raw_value
        completed = np.where(visited, q, mixed_value)
        if rescale_values:
            legal_completed = completed[mask]
            min_value, max_value = legal_completed.min(), legal_completed.max()
            completed = (completed - min_value) / max(max_value - min_value, 1e-8)
        return (maxvisit_init + n.max()) * value_scale * completed

    def _considered_argmax(self, considered_visit, completed_q):
        """argmax of seq_halving.score_considered followed by masked_argmax's masking."""
        score = np.maximum(_LOW_LOGIT, self.gumbel + self.logits + completed_q)
        score = np.where(self.n == considered_visit, score, -np.inf)
        return int(np.argmax(np.where(self.mask, score, -np.inf)))

    def select_root(self, considered_visit, maxvisit_init, value_scale, rescale_values):
        return self._considered_argmax(considered_visit, self._completed_qvalues(maxvisit_init, value_scale, rescale_values))

    def select_interior(self, maxvisit_init, value_scale, rescale_values):
        improved_policy = _softmax_legal(self.logits + self._completed_qvalues(maxvisit_init, value_scale, rescale_values), self.mask)
        to_argmax = improved_policy - self.n / (1. + self.n.sum())
        return int(np.argmax(np.where(self.mask, to_argmax, -np.inf)))

    def final(self, maxvisit_init, value_scale, rescale_values):
        completed_q = self._completed_qvalues(maxvisit_init, value_scale, rescale_values)
        action = self._considered_argmax(self.n.max(), completed_q)
        return action, _softmax_legal(self.logits + completed_q, self.mask).tobytes()

    def backup(self, action, value):
        self.w[action] += value[self.seat]
        self.n[action] += 1
        self.q[action] = self.w[action] / self.n[action]
        self.visits += 1

    def stats(self):
        return self.n.tolist(), self.q.tolist(), self.prior.tolist(), self.logits.tolist()


def _softmax_legal(logits, mask):
    x = np.where(mask, logits, -np.inf)
    x = x - x[mask].max()
    e = np.where(mask, np.exp(x), 0.)
    return e / e.sum()


_TERMINAL_VALUES = dict()


def heuristic_win_shares(game, temperature=1.):
    """Win shares (absolute seats) from the rule bots' position evaluation: softmax over the players of
    splendor.bots.evaluate_position / temperature (temperature fitted on rule-bot games, auto_research R004).
    A player who has already reached 15 points (evaluated as WIN_VALUE + points) gets everything."""
    scores = [evaluate_position(game, seat) for seat in range(N_PLAYERS)]
    top = max(scores)
    if top >= WIN_VALUE:
        winners = [1. if s >= WIN_VALUE and s == top else 0. for s in scores]
        total = sum(winners)
        return tuple(w / total for w in winners)
    e = [math.exp((s - top) / temperature) for s in scores]
    total = sum(e)
    return tuple(x / total for x in e)


def _terminal_value(env):
    """outcome of a game ended by the rules, env.final_rewards() as a tuple of floats (float32-rounded, as the
    value targets)"""
    rewards = tuple(env.game.rewards())
    value = _TERMINAL_VALUES.get(rewards)
    if value is None:
        value = _TERMINAL_VALUES[rewards] = tuple(env.final_rewards().astype(np.float64).tolist())
    return value


class SplendorMCTS:
    def __init__(self,
                 n_actions,
                 predictor,
                 n_simulation=200,
                 max_num_considered_actions=16,
                 gumbel_scale=1.,
                 value_scale=0.1,
                 maxvisit_init=50.,
                 rescale_values=True,
                 rollout_lambda=0.,
                 rollout_policy='heuristic',
                 heuristic_leaf_weight=0.,
                 heuristic_temperature=1.,
                 max_search_depth=64,
                 interrupt_callback=None,
                 seed=None,
                 node_impl='auto',
                 opponent_policy='max_n'):
        """
        predictor(request bytes) -> reply bytes (see module docstring)
        gumbel_scale: 1 for self-play (sampled move + exploration), 0 for greedy evaluation
        value_scale, maxvisit_init: sigma(q) = (maxvisit_init + max N) * value_scale * q  (mctx defaults 0.1, 50)
        rollout_lambda: λ in V = (1 - λ) v_θ + λ z_rollout; 0 = no rollout (AlphaGo Zero)
        rollout_policy: 'heuristic' | 'random' (splendor.bots), the fast rollout policy p_π
        heuristic_leaf_weight: μ in V = (1 - μ) v_θ + μ h, h = softmax over the players of the lookahead bot's
            evaluate_position / heuristic_temperature (heuristic_win_shares); 0 = network value only
        node_impl: 'auto' (C if built), 'c' or 'python'
        opponent_policy: how the tree moves at the decision nodes of the other players ([splendor]):
            'max_n'  every node selects by its own acting player's values (default, mctx interior selection)
            'prior'  the other players play the argmax of their prior (an opponent model, no branching), so the
                     simulations reach the searching player's next decisions: depth is spent on own moves
            'bot:<policy>'  the other players play that rule bot (diagnostic: an exact opponent model when the
                     opponents are that bot)
            Decisions of the searching player below the root use the usual interior selection.
        """
        assert n_simulation >= 1
        assert max_num_considered_actions >= 1
        assert 0. <= rollout_lambda <= 1., f'rollout_lambda must be in [0, 1], got {rollout_lambda}'
        assert rollout_policy in ('heuristic', 'random'), f'unknown rollout_policy {rollout_policy}'
        assert 0. <= heuristic_leaf_weight <= 1., f'heuristic_leaf_weight must be in [0, 1], got {heuristic_leaf_weight}'
        assert heuristic_temperature > 0.
        opponent_policy = opponent_policy or 'max_n'
        if opponent_policy not in ('max_n', 'prior'):
            kind, _, bot = str(opponent_policy).partition(':')
            assert kind == 'bot' and bot in ('random', 'heuristic', 'lookahead'), f'unknown opponent_policy {opponent_policy}'
        self.opponent_policy = opponent_policy
        self._opponent_bot = opponent_policy.partition(':')[2] if opponent_policy.startswith('bot:') else None
        self._opponent_action = dict()   # id(node) -> the opponent model's action at that node (fixed per search)
        self.n_actions = n_actions
        self.predictor = predictor
        self.n_simulation = n_simulation
        self.max_num_considered_actions = max_num_considered_actions
        self.gumbel_scale = gumbel_scale
        self.value_scale = value_scale
        self.maxvisit_init = maxvisit_init
        self.rescale_values = rescale_values
        self.rollout_lambda = rollout_lambda
        self.rollout_policy = rollout_policy
        self.heuristic_leaf_weight = float(heuristic_leaf_weight)
        self.heuristic_temperature = float(heuristic_temperature)
        self.max_search_depth = max_search_depth
        self.interrupt_callback = interrupt_callback
        if node_impl == 'c' or (node_impl == 'auto' and CNode is not None):
            assert CNode is not None, 'rl._mcts_core is not built (python setup_mcts_core.py build_ext --inplace)'
            self.Node = CNode
        else:
            self.Node = PyNode
        # chance draws, rollouts and Gumbel noise of the search; independent of the real game's RNG
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.considered_visit_table = get_table_of_considered_visits(max_num_considered_actions, n_simulation)

        self.root = None
        self.num_model_calls = 0
        self._cache = dict()

    # ------------------------------------------------------------------ evaluation
    def _new_node(self, env, key, reply, mask):
        """decision node of env's state from the network reply; remembered for transpositions"""
        node = self.Node(env.acting_player, reply, mask)
        self.num_model_calls += 1
        self._cache[key] = node
        return node

    def _leaf_value(self, node, env):
        """Leaf evaluation: v_θ(s_L), mixed with the rule bots' position evaluation h(s_L) (heuristic_leaf_weight μ:
        (1 - μ) v_θ + μ h) and then with a rollout outcome z_L as in AlphaGo ((1 - λ) V + λ z_L)."""
        value = node.value
        mu = self.heuristic_leaf_weight
        if mu > 0.:
            h = heuristic_win_shares(env.game, self.heuristic_temperature)
            value = tuple((1. - mu) * v + mu * hi for v, hi in zip(value, h))
        if self.rollout_lambda <= 0.:
            return value
        z = self._rollout(env)
        if z is None:   # the rollout hit max_turns: no outcome
            return value
        lam = self.rollout_lambda
        return tuple((1. - lam) * v + lam * zi for v, zi in zip(value, z))

    def _rollout(self, env):
        """One fast rollout from env's state to the end of the game, outside the tree (env is consumed):
        every phase in which a player decides (action, noble choice, returning tokens) is played by the
        rollout policy, card draws are sampled. -> win shares, or None when it hits max_turns."""
        game = env.game
        while game.phase in ('action', 'noble', 'discard') and game.turn < env.max_turns:
            game.step(bot_choose_action(game, self.rng, self.rollout_policy), self.rng, validate=False)
        if game.phase == 'terminal':
            return tuple(float(r) for r in game.rewards())
        return None

    # ------------------------------------------------------------------ search
    def simulate(self, env):
        """Search from env's current state with self.predictor.

        Returns (action_bin, policy_target) or None on interrupt:
            action_bin:    the action selected by Sequential Halving with Gumbel (the move to play)
            policy_target: pi' = softmax(logits + sigma(completed_q)) over all actions (0 for illegal), float64
        """
        search = self.simulate_iter(env)
        try:
            _, request = next(search)
            while True:
                _, request = search.send(self.predictor(request))
        except StopIteration as stop:
            return stop.value

    def simulate_iter(self, env, tag=None):
        """simulate() as a generator, for evaluating many searches in one batch: yields (tag, request bytes)
        whenever the search needs a network evaluation and expects the reply bytes to be sent back;
        returns simulate()'s result. Each simulation: select a path to a leaf, evaluate the leaf if it is
        a new state, back the value up along the path."""
        assert not env.terminated and not env.at_chance_node, 'search must start at a decision state'
        self._cache = dict()
        self._opponent_action = dict()
        self.num_model_calls = 0
        mask = env.legal_mask_bytes()
        root = self.root = self._new_node(env, env.key(), (yield tag, env.observation_bytes() + mask), mask)
        root.set_gumbel(self.gumbel_scale * self.np_rng.gumbel(size=self.n_actions))
        # count and sum of the backed-up value vectors (absolute seats) per root action: root_search_value is
        # the search's value of the played move for every player (TD(λ) value targets, train_splendor)
        root_value_sums = dict()
        self.root_search_value = None

        for _ in range(self.n_simulation):
            if self.interrupt_callback is not None and self.interrupt_callback():
                return None
            # private copy of the root state; chance draws are chosen by the search, not by env.rng
            sim_env = env.copy_for_search()
            path, value, children, child_key, key = self._select(root, sim_env)
            if value is None:
                # expansion + evaluation of the new leaf
                mask = sim_env.legal_mask_bytes()
                child = self._new_node(sim_env, key, (yield tag, sim_env.observation_bytes() + mask), mask)
                children[child_key] = child
                value = self._leaf_value(child, sim_env)
            for node, action_bin in reversed(path):
                node.backup(action_bin, value)
            sums = root_value_sums.get(path[0][1])
            if sums is None:
                root_value_sums[path[0][1]] = [1, np.asarray(value, dtype=np.float64)]
            else:
                sums[0] += 1
                sums[1] = sums[1] + np.asarray(value, dtype=np.float64)

        action_bin, policy_target = root.final(self.maxvisit_init, self.value_scale, self.rescale_values)
        sums = root_value_sums.get(action_bin)
        self.root_search_value = (sums[1] / sums[0] if sums is not None
                                  else np.asarray(root.value, dtype=np.float64))
        return action_bin, np.frombuffer(policy_target, dtype=np.float64)

    def _opponent_move(self, node, env):
        """the opponent model's action at decision node `node` (env is at its state), the same for every simulation"""
        action = self._opponent_action.get(id(node))
        if action is None:
            if self._opponent_bot is None:
                action = int(np.argmax(np.asarray(node.stats()[2])))   # the prior is 0 on illegal actions
            else:
                action = action_to_index(env.game, bot_choose_action(env.game, self.rng, self._opponent_bot))
            self._opponent_action[id(node)] = action
        return action

    def _select(self, node, env):
        """Walk down from the root to the end of one simulation.

        Returns (path [(node, action)], value, children, child_key, key):
            value is the value to back up, or None when the simulation reached a state that is new to the
            search: then children[child_key] is where its node goes and key is its transposition key."""
        path = []
        depth = 0
        own_seat = node.seat
        while True:
            if depth == 0:
                num_considered = node.num_legal
                if num_considered > self.max_num_considered_actions:
                    num_considered = self.max_num_considered_actions
                considered_visit = self.considered_visit_table[num_considered][node.visits]
                action_bin = node.select_root(considered_visit, self.maxvisit_init, self.value_scale, self.rescale_values)
            elif self.opponent_policy != 'max_n' and node.seat != own_seat:
                action_bin = self._opponent_move(node, env)
            else:
                action_bin = node.select_interior(self.maxvisit_init, self.value_scale, self.rescale_values)
            env.step(action_bin, sample_chance=False, check=False)   # legal: chosen under this state's mask
            path.append((node, action_bin))

            if env.at_chance_node:
                children = node.children
                chance = children.get(action_bin)
                if chance is None:
                    chance = children[action_bin] = dict()   # chance node: drawn card -> decision node
                pool = env.chance_pool()
                n = len(pool)
                card_id = self.rng.choices(pool, weights=[1 / n] * n)[0]
                env.resolve_chance(card_id)
                children, child_key = chance, card_id
            else:
                children, child_key = node.children, action_bin

            truncated = env.truncated
            if truncated is False and env.game.phase == 'terminal':
                # game ended by the rules inside the tree: true outcome
                return path, _terminal_value(env), None, None, None
            child = children.get(child_key)
            if child is None:
                key = env.key()
                prototype = self._cache.get(key)   # transpositions reach the same state by different paths
                if prototype is None:
                    return path, None, children, child_key, key
                child = children[child_key] = prototype.clone_fresh()
                return path, self._leaf_value(child, env), None, None, None
            if truncated or depth + 1 >= self.max_search_depth:
                # truncated at max_turns: no outcome, the network's value of the state (never expanded further)
                return path, self._leaf_value(child, env), None, None, None
            node = child
            depth += 1

    @property
    def root_visits(self):
        return None if self.root is None else np.asarray(self.root.stats()[0])

    @property
    def root_q(self):
        return None if self.root is None else np.asarray(self.root.stats()[1])
