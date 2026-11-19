"""同步高阶 SIR、Monte Carlo 及 Episode 内固定 CRN 的配对奖励。"""

import argparse
from dataclasses import dataclass
from itertools import product
import json

import numpy as np

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        return lambda function: function


def _binary_mask(graph, alive_mask_2):
    mask = np.ones(graph.M2, dtype=bool) if alive_mask_2 is None else np.asarray(alive_mask_2)
    if mask.shape != (graph.M2,) or mask.dtype != np.bool_:
        raise ValueError("alive_mask_2 必须为 bool [M2]")
    return mask


@njit(cache=True)
def _advance(state, binary, triples, beta1, beta2, mu, infection, recovery, history):
    """唯一的同步状态更新内核；标量与批量 MC 共用，随机场始终按时间/节点索引。"""
    peak = np.count_nonzero(state == 1)
    events = susceptible_steps = 0
    increment_sum = 0.0
    steps = 0
    for step in range(len(infection)):
        m2 = np.zeros(len(state), dtype=np.int64)
        m3 = np.zeros(len(state), dtype=np.int64)
        for i, j in binary:
            m2[i] += state[j] == 1
            m2[j] += state[i] == 1
        for i, j, k in triples:
            m3[i] += (state[j] == 1) and (state[k] == 1)
            m3[j] += (state[i] == 1) and (state[k] == 1)
            m3[k] += (state[i] == 1) and (state[j] == 1)
        # 计数来自旧状态；先感染与先恢复都不能影响同一步的接触数。
        for node in range(len(state)):
            if state[node] == 0:
                s2, s3 = (1-beta1)**m2[node], (1-beta2)**m3[node]
                susceptible_steps += 1
                events += m3[node] > 0
                increment_sum += s2 * (1-s3)
                if infection[step, node] < 1-s2*s3:
                    state[node] = 1
            elif state[node] == 1 and recovery[step, node] < mu:
                state[node] = 2
        count = np.count_nonzero(state == 1)
        peak = max(peak, count)
        steps = step + 1
        if len(history):
            history[step] = state
        if count == 0:
            break
    return peak, events, susceptible_steps, increment_sum, steps


@njit(cache=True)
def _batch(initial, binary, triples, beta1, beta2, mu, infection, recovery):
    values = np.empty((len(initial), 6), dtype=np.float64)
    for trajectory in range(len(initial)):
        state = initial[trajectory].copy()
        peak, events, susceptible, increment, _ = _advance(
            state, binary, triples, beta1, beta2, mu, infection[trajectory], recovery[trajectory],
            np.empty((0, 0), dtype=np.uint8))
        values[trajectory, 0] = np.count_nonzero(state != 0) / len(state)
        values[trajectory, 1] = peak / len(state)
        values[trajectory, 2] = events
        values[trajectory, 3] = susceptible
        values[trajectory, 4] = increment
        values[trajectory, 5] = np.count_nonzero(state == 1) == 0
    return values


@dataclass(frozen=True)
class SIRParameters:
    beta1: float
    beta2: float
    mu: float
    rho0: float
    max_steps: int = 2000

    def __post_init__(self):
        for name in ("beta1", "beta2", "mu", "rho0"):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} 必须为 [0,1] 内的有限数值")
        if not isinstance(self.max_steps, int) or self.max_steps < 1:
            raise ValueError("max_steps 必须为正整数")


def _infection_probabilities(binary_edges, triples, infected, beta1, beta2):
    m2 = np.zeros(len(infected), dtype=np.int64)
    i, j = binary_edges.T
    np.add.at(m2, i, infected[j])
    np.add.at(m2, j, infected[i])
    m3 = np.zeros(len(infected), dtype=np.int64)
    edge_infected = infected[triples]
    for position, others in ((0, (1, 2)), (1, (0, 2)), (2, (0, 1))):
        np.add.at(m3, triples[:, position],
                  edge_infected[:, others[0]] & edge_infected[:, others[1]])
    binary_survival = (1 - beta1) ** m2
    triple_survival = (1 - beta2) ** m3
    probability = 1 - binary_survival * triple_survival
    # 记录概率增量，不能将条件激活误称为已经发生的三元因果感染。
    increment = binary_survival * (1 - triple_survival)
    return probability, m3, increment


def simulate_sir(graph, parameters, alive_mask=None, *, seed, episode_id=0,
                 trajectory_id=0, initial_infected=None, return_history=False, alive_mask_2=None):
    """运行单条轨迹。状态 0/1/2 分别为 S/I/R；未收敛时 F/P 为 None。

    initial_infected 仅用于指定初始集合的机制检查；默认按 rho0 均匀无放回抽取。
    每步各通道始终为全部 N 个节点抽样，事件数量不会改变随机数索引。
    """
    ids = graph.alive_edge_ids(alive_mask)
    triples = graph.hyperedge_nodes_3[ids]
    binary = graph.edge_index_2[_binary_mask(graph, alive_mask_2)]
    seeds = np.random.SeedSequence([seed, 2, episode_id, trajectory_id]).spawn(3)
    initial_rng, infection_rng, recovery_rng = map(np.random.default_rng, seeds)
    if initial_infected is None:
        count = max(1, round(parameters.rho0 * graph.N))
        initial = initial_rng.choice(graph.N, count, replace=False)
    else:
        initial = np.asarray(initial_infected)
        if (initial.ndim != 1 or not len(initial)
                or not np.issubdtype(initial.dtype, np.integer)
                or np.any(initial < 0) or np.any(initial >= graph.N)
                or len(np.unique(initial)) != len(initial)):
            raise ValueError("初始感染集合必须包含互异且有效的节点 ID")
    state = np.zeros(graph.N, dtype=np.uint8)
    state[initial] = 1
    peak = len(initial)
    history = [state.copy()] if return_history else None
    activation_events = susceptible_node_steps = 0
    increment_sum = 0.0
    steps = 0
    while steps < parameters.max_steps:
        size = min(64, parameters.max_steps - steps)
        block_history = np.empty((size if return_history else 0, graph.N), dtype=np.uint8)
        p, events, susceptible, increment, used = _advance(
            state, binary, triples, parameters.beta1, parameters.beta2, parameters.mu,
            infection_rng.random((size, graph.N)), recovery_rng.random((size, graph.N)), block_history)
        peak = max(peak, p)
        activation_events += events
        susceptible_node_steps += susceptible
        increment_sum += increment
        steps += used
        if return_history:
            history.extend(block_history[:used].copy())
        if not np.any(state == 1):
            break
    converged = not np.any(state == 1)
    ever_fraction = float(np.count_nonzero(state != 0) / graph.N)
    return {
        "converged": bool(converged), "steps": steps,
        "F": ever_fraction if converged else None,
        "P": peak / graph.N if converged else None,
        "ever_infected_fraction_so_far": ever_fraction,
        "peak_so_far": peak / graph.N,
        "ho_activated": activation_events > 0,
        "ho_activation_events": activation_events,
        "ho_probability_increment_sum": increment_sum,
        "susceptible_node_steps": susceptible_node_steps,
        "final_state": state,
        "history": np.asarray(history) if return_history else None,
    }


class SIRMonteCarlo:
    """绑定一张图和固定动力学；同一 Episode 复用随机场及 Mask 结果缓存。

    每个轨迹 ID 独立派生初始/感染/恢复种子，按需重建随机流而非存储大随机场。
    新 Episode 调用 reset() 或显式传入新的 episode_id，避免反复拟合同一批随机数。
    """

    def __init__(self, graph, parameters, *, seed, mc, episode_id=0, batch=False):
        if not isinstance(mc, int) or mc < 1:
            raise ValueError("mc 必须为正整数")
        if not isinstance(episode_id, int) or episode_id < 0:
            raise ValueError("episode_id 必须为非负整数")
        self.graph, self.parameters = graph, parameters
        self.seed, self.mc, self.episode_id = seed, mc, episode_id
        self._cache = {}
        self.trajectory_calls = 0
        self.batch = batch
        self._random_field = None

    def reset(self, episode_id=None):
        episode_id = self.episode_id + 1 if episode_id is None else episode_id
        if not isinstance(episode_id, int) or episode_id < 0:
            raise ValueError("episode_id 必须为非负整数")
        self.episode_id = episode_id
        self._cache.clear()
        self._random_field = None

    def estimate(self, alive_mask=None, *, alive_mask_2=None):
        mask = np.ones(self.graph.M0, dtype=bool) if alive_mask is None else np.asarray(alive_mask)
        self.graph.alive_edge_ids(mask)
        mask2 = _binary_mask(self.graph, alive_mask_2)
        key = (mask2.tobytes(), mask.tobytes())
        if key not in self._cache:
            if self.batch:
                results = self._batch_results(mask, mask2)
            else:
                results = [simulate_sir(self.graph, self.parameters, mask, seed=self.seed,
                                        episode_id=self.episode_id, trajectory_id=i, alive_mask_2=mask2)
                           for i in range(self.mc)]
            self.trajectory_calls += self.mc
            converged = np.asarray([r["converged"] for r in results])
            F = np.asarray([r["F"] for r in results], dtype=float)
            P = np.asarray([r["P"] for r in results], dtype=float)
            complete = bool(converged.all())
            susceptible_steps = sum(r["susceptible_node_steps"] for r in results)
            summary = {
                "status": "CONVERGED" if complete else "NOT_CONVERGED",
                "episode_id": self.episode_id, "mc": self.mc,
                # 不剔除截断轨迹后偷偷求均值，以免改变估计分布。
                "F": float(F.mean()) if complete else None,
                "P": float(P.mean()) if complete else None,
                "F_standard_error": float(F.std(ddof=1) / np.sqrt(self.mc))
                if complete and self.mc > 1 else None,
                "F_samples": F, "P_samples": P, "converged": converged,
                "unconverged_count": int(np.count_nonzero(~converged)),
                "unconverged_fraction": float(np.mean(~converged)),
                "ho_activation_fraction": float(np.mean([r["ho_activated"] for r in results])),
                "ho_probability_increment_mean":
                sum(r["ho_probability_increment_sum"] for r in results) / susceptible_steps
                if susceptible_steps else None,
            }
            for array in (F, P, converged):
                array.setflags(write=False)
            self._cache[key] = summary
        return dict(self._cache[key])

    def _batch_results(self, mask, mask2):
        # 只缓存前 64 步随机场；长轨迹由同一内核按原始种子完整重算，不截断。
        g, p = self.graph, self.parameters
        if self._random_field is None:
            initial = np.zeros((self.mc, g.N), dtype=np.uint8)
            infection = np.empty((self.mc, min(64, p.max_steps), g.N))
            recovery = np.empty_like(infection)
            for i in range(self.mc):
                seeds = np.random.SeedSequence([self.seed, 2, self.episode_id, i]).spawn(3)
                ir, inf, rec = map(np.random.default_rng, seeds)
                initial[i, ir.choice(g.N, max(1, round(p.rho0*g.N)), replace=False)] = 1
                infection[i] = inf.random(infection.shape[1:])
                recovery[i] = rec.random(recovery.shape[1:])
            self._random_field = initial, infection, recovery
        initial, infection, recovery = self._random_field
        values = _batch(initial, g.edge_index_2[mask2], g.hyperedge_nodes_3[mask],
                        p.beta1, p.beta2, p.mu, infection, recovery)
        results = []
        for i, (F, P, events, susceptible, increment, complete) in enumerate(values):
            if not complete:
                result = simulate_sir(g, p, mask, seed=self.seed, episode_id=self.episode_id,
                                      trajectory_id=i, alive_mask_2=mask2)
            else:
                result = dict(converged=True, F=F, P=P, ho_activated=events > 0,
                              ho_probability_increment_sum=increment, susceptible_node_steps=susceptible)
            results.append(result)
        return results

    def paired_reward(self, before_mask, after_mask, *, before_mask_2=None, after_mask_2=None):
        """配对估计 ΔF；只在全部轨迹收敛时返回有效奖励，不裁剪负样本。"""
        before = self.estimate(before_mask, alive_mask_2=before_mask_2)
        after = self.estimate(after_mask, alive_mask_2=after_mask_2)
        differences = before["F_samples"] - after["F_samples"]
        differences.setflags(write=False)
        complete = before["F"] is not None and after["F"] is not None
        variance = float(differences.var(ddof=1)) if complete and self.mc > 1 else None
        standard_error = np.sqrt(variance / self.mc) if variance is not None else None
        return {
            "status": "CONVERGED" if complete else "NOT_CONVERGED",
            "F_before": before["F"], "F_after": after["F"],
            # 直接相减缓存的 F，使同一 Episode 的奖励累计满足望远镜性质。
            "reward": before["F"] - after["F"] if complete else None,
            "difference_samples": differences, "paired_variance": variance,
            "standard_error": float(standard_error) if standard_error is not None else None,
            "ci95_half_width": float(1.96 * standard_error) if standard_error is not None else None,
        }


if __name__ == "__main__":
    from .generators import generate_graph
    from .utils import load_config

    parser = argparse.ArgumentParser(description="单图、小规模 SIR 候选参数试算")
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mc", type=int, default=None)
    parser.add_argument("--beta1", type=float, nargs="+", default=None)
    parser.add_argument("--beta2", type=float, nargs="+", default=None)
    parser.add_argument("--rho0", type=float, nargs="+", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    p = config["sir"]
    seed = config["seed"] if args.seed is None else args.seed
    mc = p["calibration_mc"] if args.mc is None else args.mc
    graph, _ = generate_graph(config, hyperedge="Overlap", seed=seed)
    for beta1, beta2, rho0 in product(args.beta1 or [p["beta1"]],
                                     args.beta2 or [p["beta2"]], args.rho0 or [p["rho0"]]):
        parameters = SIRParameters(beta1, beta2, p["mu"], rho0, p["max_steps"])
        estimator = SIRMonteCarlo(graph, parameters, seed=seed, mc=mc)
        full = estimator.estimate()
        all_alive, no_triples = np.ones(graph.M0, dtype=bool), np.zeros(graph.M0, dtype=bool)
        delta_ho = estimator.paired_reward(all_alive, no_triples)
        actions = {}
        for edge_id in range(min(2, graph.M0)):
            mask = all_alive.copy()
            mask[edge_id] = False
            actions[edge_id] = {key: value for key, value in estimator.paired_reward(all_alive, mask).items()
                                if key != "difference_samples"}
        print(json.dumps({"graph_id": graph.graph_id, "seed": seed, "mc": mc,
                          "parameters": {"beta1": beta1, "beta2": beta2, "mu": p["mu"],
                                         "rho0": rho0, "max_steps": p["max_steps"]},
                          "full": {key: value for key, value in full.items()
                                   if key not in ("F_samples", "P_samples", "converged")},
                          "delta_HO": delta_ho["reward"], "candidate_rewards": actions,
                          "science_status": "NOT_EVALUATED"}, ensure_ascii=False, indent=2))
