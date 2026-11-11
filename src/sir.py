"""同步高阶 SIR、Monte Carlo 及 Episode 内固定 CRN 的配对奖励。"""

import argparse
from dataclasses import dataclass
from itertools import product
import json

import numpy as np


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
                 trajectory_id=0, initial_infected=None, return_history=False):
    """运行单条轨迹。状态 0/1/2 分别为 S/I/R；未收敛时 F/P 为 None。

    initial_infected 仅用于指定初始集合的机制检查；默认按 rho0 均匀无放回抽取。
    每步各通道始终为全部 N 个节点抽样，事件数量不会改变随机数索引。
    """
    ids = graph.alive_edge_ids(alive_mask)
    triples = graph.hyperedge_nodes_3[ids]
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
    for steps in range(1, parameters.max_steps + 1):
        infected, susceptible = state == 1, state == 0
        probability, m3, increment = _infection_probabilities(
            graph.edge_index_2, triples, infected, parameters.beta1, parameters.beta2
        )
        activation_events += int(np.count_nonzero(susceptible & (m3 > 0)))
        susceptible_node_steps += int(np.count_nonzero(susceptible))
        increment_sum += float(increment[susceptible].sum())
        new_infected = susceptible & (infection_rng.random(graph.N) < probability)
        recovered = infected & (recovery_rng.random(graph.N) < parameters.mu)
        # 两类事件都由旧状态计算，新感染者不能在同一步恢复或继续传染。
        state[new_infected] = 1
        state[recovered] = 2
        infected_count = int(np.count_nonzero(state == 1))
        peak = max(peak, infected_count)
        if return_history:
            history.append(state.copy())
        if infected_count == 0:
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

    def __init__(self, graph, parameters, *, seed, mc, episode_id=0):
        if not isinstance(mc, int) or mc < 1:
            raise ValueError("mc 必须为正整数")
        if not isinstance(episode_id, int) or episode_id < 0:
            raise ValueError("episode_id 必须为非负整数")
        self.graph, self.parameters = graph, parameters
        self.seed, self.mc, self.episode_id = seed, mc, episode_id
        self._cache = {}
        self.trajectory_calls = 0

    def reset(self, episode_id=None):
        episode_id = self.episode_id + 1 if episode_id is None else episode_id
        if not isinstance(episode_id, int) or episode_id < 0:
            raise ValueError("episode_id 必须为非负整数")
        self.episode_id = episode_id
        self._cache.clear()

    def estimate(self, alive_mask=None):
        mask = np.ones(self.graph.M0, dtype=bool) if alive_mask is None else np.asarray(alive_mask)
        self.graph.alive_edge_ids(mask)
        key = mask.tobytes()
        if key not in self._cache:
            results = [simulate_sir(self.graph, self.parameters, mask, seed=self.seed,
                                    episode_id=self.episode_id, trajectory_id=i)
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

    def paired_reward(self, before_mask, after_mask):
        """配对估计 ΔF；只在全部轨迹收敛时返回有效奖励，不裁剪负样本。"""
        before, after = self.estimate(before_mask), self.estimate(after_mask)
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
