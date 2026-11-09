"""Robot seats: the trained agent (network + deployment search) or a rule bot. choose(game) -> engine Action for the
current player of `game` in any decision phase (action, discard, noble)."""

from __future__ import annotations

import os
import random
import sys
import threading

from ..bots import choose_action

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_WEIGHTS = os.path.join(ROOT, "models", "splendor_final", "weights.pth")
DEFAULT_CONFIG = os.path.join(ROOT, "models", "splendor_final", "config.yml")

POLICY_LABELS = {
    "model": "训练模型（网络 + 搜索）",
    "lookahead": "规则机器人 lookahead",
    "heuristic": "规则机器人 heuristic",
    "random": "随机",
}


class RuleRobot:
    def __init__(self, policy, seed=None):
        self.policy = policy
        self.rng = random.Random(seed)

    def choose(self, game):
        return choose_action(game, self.rng, self.policy)


class ModelProvider:
    """loads the agent once (lazily, thread-safe) and shares it between the robot seats"""

    def __init__(self, weights=DEFAULT_WEIGHTS, config=DEFAULT_CONFIG, device=None, sims=512, considered=8):
        self.weights, self.config, self.device = weights, config, device
        self.sims, self.considered = sims, considered
        self._agent = None
        self._lock = threading.Lock()

    def agent(self):
        with self._lock:
            if self._agent is None:
                if ROOT not in sys.path:
                    sys.path.insert(0, ROOT)
                import torch
                from models.splendor_final.play_agent import Agent
                device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
                self._agent = Agent(weights=self.weights, config=self.config, device=device, sims=self.sims,
                                    considered=self.considered, opponent_policy="prior")
                self.device = device
            return self._agent

    def describe(self):
        return {"weights": os.path.relpath(self.weights, ROOT), "sims": self.sims, "considered": self.considered,
                "device": self.device or "未加载", "loaded": self._agent is not None}


class ModelRobot:
    def __init__(self, provider):
        self.provider = provider
        self.lock = threading.Lock()

    def choose(self, game):
        from ..rl_env import SplendorRLEnv, index_to_action
        agent = self.provider.agent()
        env = SplendorRLEnv(max_turns=max(400, game.turn + 400))
        env.game = game.clone()
        with self.lock:
            index = agent.act(env)
        return index_to_action(env.game, index)


def make_robot(policy, provider=None, seed=None):
    if policy == "model":
        if provider is None:
            raise ValueError("没有可用的训练模型")
        return ModelRobot(provider)
    if policy in ("lookahead", "heuristic", "random", "lookahead_alt1", "lookahead_alt2"):
        return RuleRobot(policy, seed)
    raise ValueError(f"未知的机器人策略：{policy}")
