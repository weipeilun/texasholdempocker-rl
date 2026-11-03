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
    * 4 players: values are absolute-seat vectors; every node uses its own acting player's
      component (max^n), no alternating negation.
"""

import math
import random

import numpy as np

from splendor.bots import choose_action as bot_choose_action
from splendor.rl_env import to_absolute, N_PLAYERS

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


def get_table_of_considered_visits(max_num_considered_actions, num_simulations):
    return tuple(get_sequence_of_considered_visits(m, num_simulations)
                 for m in range(max_num_considered_actions + 1))


# ---------------------------------------------------------------------------- tree
class _DecisionNode:
    __slots__ = ('seat', 'logits', 'prior', 'mask', 'value', 'children', 'w', 'n', 'q')

    def __init__(self, n_actions, seat, prior, mask, value):
        self.seat = seat                 # acting player of this state
        self.mask = mask                 # legal actions
        self.prior = prior               # P(s, .), masked and normalised
        # prior logits, max-normalised over legal actions; illegal = -inf (only used under the mask)
        logits = np.where(mask, np.log(np.maximum(prior, _TINY)), -np.inf)
        self.logits = logits - logits[mask].max()
        self.value = value               # v_θ(s), absolute-seat vector
        self.children = dict()           # action -> _DecisionNode | _ChanceNode
        self.w = np.zeros(n_actions, dtype=np.float64)
        self.n = np.zeros(n_actions, dtype=np.float64)
        self.q = np.zeros(n_actions, dtype=np.float64)   # from self.seat's perspective


class _ChanceNode:
    __slots__ = ('children',)

    def __init__(self):
        self.children = dict()  # drawn card id -> _DecisionNode


def _softmax_legal(logits, mask):
    x = np.where(mask, logits, -np.inf)
    x = x - x[mask].max()
    e = np.where(mask, np.exp(x), 0.)
    return e / e.sum()


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
                 max_search_depth=64,
                 interrupt_callback=None,
                 seed=None):
        """
        predictor(observation, legal_mask) -> (action_probs[n_actions], values_relative[n_players])
        gumbel_scale: 1 for self-play (sampled move + exploration), 0 for greedy evaluation
        value_scale, maxvisit_init: sigma(q) = (maxvisit_init + max N) * value_scale * q  (mctx defaults 0.1, 50)
        rollout_lambda: λ in V = (1 - λ) v_θ + λ z_rollout; 0 = no rollout (AlphaGo Zero)
        rollout_policy: 'heuristic' | 'random' (splendor.bots), the fast rollout policy p_π
        """
        assert n_simulation >= 1
        assert max_num_considered_actions >= 1
        assert 0. <= rollout_lambda <= 1., f'rollout_lambda must be in [0, 1], got {rollout_lambda}'
        assert rollout_policy in ('heuristic', 'random'), f'unknown rollout_policy {rollout_policy}'
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
        self.max_search_depth = max_search_depth
        self.interrupt_callback = interrupt_callback
        # chance draws, rollouts and Gumbel noise of the search; independent of the real game's RNG
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.considered_visit_table = get_table_of_considered_visits(max_num_considered_actions, n_simulation)

        self.root = None
        self.root_gumbel = None
        self.num_model_calls = 0
        self._cache = dict()

    # ------------------------------------------------------------------ evaluation
    def _expand(self, env):
        """New decision node for env's (non-terminal, non-chance) state: one model evaluation."""
        key = env.key()
        cached = self._cache.get(key)   # transpositions reach the same state by different paths
        if cached is None:
            mask = env.legal_mask()
            probs, values_rel = self.predictor(env.observation(), mask)
            self.num_model_calls += 1
            probs = np.asarray(probs, dtype=np.float64) * mask
            total = probs.sum()
            probs = probs / total if total > 0 else mask / mask.sum()
            cached = (probs, mask, to_absolute(values_rel, env.acting_player).astype(np.float64))
            self._cache[key] = cached
        probs, mask, value = cached
        return _DecisionNode(self.n_actions, env.acting_player, probs, mask, value)

    def _leaf_value(self, node, env):
        """AlphaGo leaf evaluation V(s_L) = (1 - λ) v_θ(s_L) + λ z_L."""
        if self.rollout_lambda <= 0.:
            return node.value
        z = self._rollout(env)
        return (1. - self.rollout_lambda) * node.value + self.rollout_lambda * z

    def _rollout(self, env):
        """One fast rollout from env's state to the end of the game, outside the tree (env is consumed)."""
        game = env.game
        while game.phase in ('action', 'noble') and game.turn < env.max_turns:
            game.step(bot_choose_action(game, self.rng, self.rollout_policy), self.rng)
        if game.phase == 'terminal':
            return np.asarray(game.rewards(), dtype=np.float64)
        return np.zeros(N_PLAYERS, dtype=np.float64)

    # ------------------------------------------------------------------ Q transform
    def _completed_qvalues(self, node):
        """sigma(completed Q) of qtransform_completed_by_mix_value; illegal entries are meaningless."""
        n, q, mask = node.n, node.q, node.mask
        visited = n > 0
        raw_value = node.value[node.seat]
        sum_n = n.sum()
        if sum_n > 0:
            prior = np.maximum(node.prior, _TINY)
            sum_probs = prior[visited].sum()
            weighted_q = (prior[visited] * q[visited]).sum() / sum_probs
            mixed_value = (raw_value + sum_n * weighted_q) / (sum_n + 1)
        else:
            mixed_value = raw_value
        completed = np.where(visited, q, mixed_value)
        if self.rescale_values:
            legal_completed = completed[mask]
            min_value, max_value = legal_completed.min(), legal_completed.max()
            completed = (completed - min_value) / max(max_value - min_value, 1e-8)
        return (self.maxvisit_init + n.max()) * self.value_scale * completed

    def _score_considered(self, node, considered_visit, completed_q):
        """seq_halving.score_considered followed by masked_argmax's masking."""
        score = np.maximum(_LOW_LOGIT, self.root_gumbel + node.logits + completed_q)
        score = np.where(node.n == considered_visit, score, -np.inf)
        return np.where(node.mask, score, -np.inf)

    # ------------------------------------------------------------------ action selection
    def _root_action(self, node):
        num_considered = min(self.max_num_considered_actions, int(node.mask.sum()))
        simulation_index = int(node.n.sum())          # at the root = number of finished simulations
        considered_visit = self.considered_visit_table[num_considered][simulation_index]
        return int(np.argmax(self._score_considered(node, considered_visit, self._completed_qvalues(node))))

    def _interior_action(self, node):
        improved_policy = _softmax_legal(node.logits + self._completed_qvalues(node), node.mask)
        to_argmax = improved_policy - node.n / (1. + node.n.sum())
        return int(np.argmax(np.where(node.mask, to_argmax, -np.inf)))

    # ------------------------------------------------------------------ search
    def simulate(self, env):
        """Search from env's current state.

        Returns (action_bin, policy_target) or None on interrupt:
            action_bin:    the action selected by Sequential Halving with Gumbel (the move to play)
            policy_target: pi' = softmax(logits + sigma(completed_q)) over all actions (0 for illegal)
        """
        assert not env.terminated and not env.at_chance_node, 'search must start at a decision state'
        self._cache = dict()
        self.num_model_calls = 0
        self.root = self._expand(env)
        self.root_gumbel = self.gumbel_scale * self.np_rng.gumbel(size=self.n_actions)

        for _ in range(self.n_simulation):
            if self.interrupt_callback is not None and self.interrupt_callback():
                return None
            # private copy of the root state; chance draws are chosen by the search, not by env.rng
            self._simulate_decision(self.root, env.new_random(), 0)

        completed_q = self._completed_qvalues(self.root)
        considered_visit = self.root.n.max()
        action_bin = int(np.argmax(self._score_considered(self.root, considered_visit, completed_q)))
        policy_target = _softmax_legal(self.root.logits + completed_q, self.root.mask)
        return action_bin, policy_target

    def _simulate_decision(self, node, env, depth):
        """One simulation through an expanded decision node; returns the backed-up value vector."""
        action_bin = self._root_action(node) if depth == 0 else self._interior_action(node)
        env.step(action_bin, sample_chance=False)

        if env.at_chance_node:
            chance = node.children.get(action_bin)
            if chance is None:
                chance = _ChanceNode()
                node.children[action_bin] = chance
            cards, probabilities = zip(*env.chance_outcomes())
            card_id = self.rng.choices(cards, weights=probabilities)[0]
            env.resolve_chance(card_id)
            value = self._descend(chance.children, card_id, env, depth)
        else:
            value = self._descend(node.children, action_bin, env, depth)

        node.w[action_bin] += value[node.seat]
        node.n[action_bin] += 1
        node.q[action_bin] = node.w[action_bin] / node.n[action_bin]
        return value

    def _descend(self, children, child_key, env, depth):
        """Value of the state env reached, stored under children[child_key]."""
        if env.terminated:
            # terminal state inside the tree: true outcome
            return env.final_rewards().astype(np.float64)
        child = children.get(child_key)
        if child is None:
            # expansion + evaluation of the new leaf
            child = self._expand(env)
            children[child_key] = child
            return self._leaf_value(child, env)
        if depth + 1 >= self.max_search_depth:
            return self._leaf_value(child, env)
        return self._simulate_decision(child, env, depth + 1)

    @property
    def root_visits(self):
        return None if self.root is None else self.root.n.copy()

    @property
    def root_q(self):
        return None if self.root is None else self.root.q.copy()
