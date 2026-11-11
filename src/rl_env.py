"""固定预算删边环境、原始结构状态与简洁经验回放。"""

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np
import torch

from .sir import SIRMonteCarlo


@dataclass(frozen=True, eq=False)
class State:
    graph_id: str
    alive_mask: np.ndarray
    remaining_budget: int
    K: int

    def __post_init__(self):
        mask = np.asarray(self.alive_mask)
        if mask.ndim != 1 or mask.dtype != np.bool_:
            raise ValueError("alive_mask 必须为 bool [M0]")
        if not isinstance(self.K, int) or not 1 <= self.K <= len(mask):
            raise ValueError("K 必须在 [1,M0] 内")
        if not isinstance(self.remaining_budget, int) or not 0 <= self.remaining_budget <= self.K:
            raise ValueError("remaining_budget 必须在 [0,K] 内")
        if np.count_nonzero(~mask) != self.K - self.remaining_budget:
            raise ValueError("存活 Mask 与已消耗预算不一致")
        mask = mask.copy()
        mask.setflags(write=False)
        object.__setattr__(self, "alive_mask", mask)

    @property
    def done(self):
        return self.remaining_budget == 0


@dataclass(frozen=True, eq=False)
class Transition:
    graph_id: str
    mask_before: np.ndarray
    action_edge_id: int
    reward: float
    mask_after: np.ndarray
    done: bool
    K: int
    episode_id: int
    sir_config_id: str

    def __post_init__(self):
        before = State(self.graph_id, self.mask_before,
                       self.K - int(np.count_nonzero(~np.asarray(self.mask_before))), self.K)
        after = State(self.graph_id, self.mask_after, before.remaining_budget - 1, self.K)
        action = self.action_edge_id
        if not isinstance(action, (int, np.integer)) or not 0 <= action < len(before.alive_mask):
            raise ValueError("动作原始 ID 无效")
        expected = before.alive_mask.copy()
        if not expected[action]:
            raise ValueError("不能重复删除超边")
        expected[action] = False
        if not np.array_equal(expected, after.alive_mask) or self.done != after.done:
            raise ValueError("Transition 必须恰好删除一个合法动作且 done 与预算一致")
        if not np.isfinite(self.reward):
            raise ValueError("奖励必须为有限标量")
        object.__setattr__(self, "mask_before", before.alive_mask)
        object.__setattr__(self, "mask_after", after.alive_mask)

    @property
    def state(self):
        return State(self.graph_id, self.mask_before,
                     self.K - int(np.count_nonzero(~self.mask_before)), self.K)

    @property
    def next_state(self):
        return State(self.graph_id, self.mask_after, self.state.remaining_budget - 1, self.K)


class DeletionEnv:
    def __init__(self, graphs, parameters, *, seed, mc, gamma=1.0):
        if gamma != 1.0:
            raise ValueError("主实验要求 gamma=1")
        self.graphs, self.parameters, self.seed, self.mc = dict(graphs), parameters, seed, mc
        self.sir_config_id = hashlib.sha256(json.dumps(asdict(parameters), sort_keys=True).encode()).hexdigest()[:16]
        self.episode_id = -1
        self.state, self.monte_carlo = None, None
        self.sir_trajectories = 0

    def reset(self, graph_id, K):
        graph = self.graphs[graph_id]
        state = State(graph_id, np.ones(graph.M0, dtype=bool), K, K)
        self.episode_id += 1
        estimator = SIRMonteCarlo(graph, self.parameters, seed=self.seed, mc=self.mc,
                                  episode_id=self.episode_id)
        estimate = estimator.estimate()
        self.sir_trajectories += estimator.trajectory_calls
        self.state, self.monte_carlo = None, None
        if estimate["F"] is None:
            raise RuntimeError("NOT_CONVERGED: reset 的 SIR 轨迹未达到吸收态")
        self.state, self.monte_carlo = state, estimator
        return state

    def step(self, edge_id):
        if self.state is None or self.state.done:
            raise ValueError("必须先 reset，且不能在终止后继续 step")
        if (not isinstance(edge_id, (int, np.integer)) or isinstance(edge_id, (bool, np.bool_))
                or not 0 <= edge_id < len(self.state.alive_mask) or not self.state.alive_mask[edge_id]):
            raise ValueError("动作必须为存活超边的原始 ID")
        mask = self.state.alive_mask.copy()
        mask[edge_id] = False
        calls_before = self.monte_carlo.trajectory_calls
        estimate = self.monte_carlo.paired_reward(self.state.alive_mask, mask)
        self.sir_trajectories += self.monte_carlo.trajectory_calls - calls_before
        if estimate["reward"] is None:
            raise RuntimeError("NOT_CONVERGED: 不提交删边状态或截断奖励")
        self.state = State(self.state.graph_id, mask, self.state.remaining_budget - 1, self.state.K)
        info = {key: value for key, value in estimate.items() if key != "difference_samples"}
        info.update(episode_id=self.episode_id, sir_config_id=self.sir_config_id)
        return self.state, estimate["reward"], self.state.done, info


class ReplayBuffer:
    def __init__(self, capacity=10000, seed=0):
        if not isinstance(capacity, int) or capacity < 1:
            raise ValueError("capacity 必须为正整数")
        self.capacity, self.position = capacity, 0
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, 8]))
        self._items = []

    def __len__(self):
        return len(self._items)

    def add(self, transition):
        if not isinstance(transition, Transition):
            raise TypeError("Replay 只保存 Transition，不保存 embedding")
        if len(self._items) < self.capacity:
            self._items.append(transition)
        else:
            self._items[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if not isinstance(batch_size, int) or not 1 <= batch_size <= len(self):
            raise ValueError("batch_size 超出当前 Replay 大小")
        return [self._items[i] for i in self.rng.choice(len(self), batch_size, replace=False)]


def action_features(encoder, graph, state):
    if state.done or graph.graph_id != state.graph_id:
        raise ValueError("动作评分需要同一张图上的非终止 State")
    z, global_z, valid = encoder(graph, state.alive_mask)
    budget = z.new_full((len(z), 1), state.remaining_budget / state.K)
    return torch.cat((z, global_z.expand(len(z), -1), budget), dim=-1), valid
