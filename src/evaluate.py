"""统一策略、独立测试 CRN、图级统计与单一结果 CSV。"""

import argparse
from collections import defaultdict
from copy import deepcopy
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from .ddqn import DDQNAgent, DeletionEnv, train, ensure_unseen, save_checkpoint
from .hgnn import ABLATIONS, ablation_config
from .hypergraph import compute_metrics
from .rl_env import State
from .sac import SACAgent
from .sir import SIRParameters, SIRMonteCarlo
from .utils import load_config


def load_agent(path, device="cpu"):
    """只读取张量及基本类型的推断 checkpoint，不恢复 Replay 或优化器。"""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    classes = {"ddqn": DDQNAgent, "sac": SACAgent}
    agent = classes[checkpoint["algorithm"]]({}, checkpoint["config"], seed=checkpoint["seed"], device=device)
    agent.load_inference_state(checkpoint["model"])
    agent.provenance = checkpoint["provenance"]
    agent.sir_parameters = SIRParameters(**checkpoint["sir_parameters"])
    agent.training_resources = checkpoint["resources"]
    (agent.online if hasattr(agent, "online") else agent.actor).eval()
    return agent, checkpoint


def _policy_info(name, policy):
    if isinstance(policy, str):
        if policy not in ("random", "overlap_degree", "sir_greedy"):
            raise ValueError("未知基线")
        return policy, "full", None
    return ("ddqn" if isinstance(policy, DDQNAgent) else "sac",
            policy.config.get("ablation", "full"), policy.seed)


def evaluate_policies(samples, policies, parameters, K, *, mc, seed, greedy_mc=128):
    """所有方法在同一图上使用同一独立测试随机组，贪心选边使用另一随机组。"""
    samples = sorted(list(samples), key=lambda sample: sample[0].graph_id)
    if not samples or not policies:
        raise ValueError("测试图和策略不能为空")
    signatures = [_policy_info(name, policy) for name, policy in policies.items()]
    if len(set(signatures)) != len(signatures):
        raise ValueError("同一算法/消融/训练种子不能重复计入统计")
    for graph, metadata in samples:
        if metadata.get("split") not in ("test_id", "ood_size", "ood_structure"):
            raise ValueError("最终评价只能使用 Test-ID/OOD 留出图")
        for policy in policies.values():
            if not isinstance(policy, str):
                if policy.config["rl"]["K"] != K or getattr(policy, "sir_parameters", parameters) != parameters:
                    raise ValueError("比较策略必须使用相同 K 与 SIR 条件")
                provenance = getattr(policy, "provenance", {})
                if metadata.get("family_id") in provenance.get("group_ids", []):
                    raise ValueError("SPLIT_LEAKAGE: 测试家族被用于训练或选模")
                ensure_unseen(graph, provenance)
    rows = []
    for graph_index, (graph, metadata) in enumerate(samples):
        test_seed = int(np.random.SeedSequence([seed, 20, graph_index]).generate_state(1)[0])
        selection_seed = int(np.random.SeedSequence([seed, 21, graph_index]).generate_state(1)[0])
        for name, policy in policies.items():
            method, ablation, training_seed = _policy_info(name, policy)
            if not isinstance(policy, str):
                (policy.online if hasattr(policy, "online") else policy.actor).eval()
            rng = np.random.default_rng(np.random.SeedSequence([seed, 22, graph_index]))
            state = State(graph.graph_id, np.ones(graph.M0, dtype=bool), K, K)
            estimator = SIRMonteCarlo(graph, parameters, seed=test_seed, mc=mc)
            greedy = SIRMonteCarlo(graph, parameters, seed=selection_seed, mc=greedy_mc) if method == "sir_greedy" else None
            initial = estimator.estimate()
            if initial["F"] is None:
                raise RuntimeError("NOT_CONVERGED: 测试初态传播未收敛")
            actions = []
            start = time.perf_counter()
            while not state.done:
                ids = np.flatnonzero(state.alive_mask)
                if method == "random":
                    action = int(rng.choice(ids))
                elif method == "overlap_degree":
                    scores = compute_metrics(graph, state.alive_mask)["o2"]
                    action = int(ids[np.argmax(scores[ids])])
                elif method == "sir_greedy":
                    values = []
                    for edge_id in ids:
                        mask = state.alive_mask.copy()
                        mask[edge_id] = False
                        reward = greedy.paired_reward(state.alive_mask, mask)["reward"]
                        if reward is None:
                            raise RuntimeError("NOT_CONVERGED: 贪心候选传播未收敛")
                        values.append(reward)
                    action = int(ids[np.argmax(values)])
                else:
                    action = policy.act(state, graph=graph, explore=False)
                if action not in ids:
                    raise ValueError("策略选择了已删除或无效超边")
                mask = state.alive_mask.copy()
                mask[action] = False
                actions.append(action)
                state = State(graph.graph_id, mask, state.remaining_budget - 1, K)
            inference_seconds = time.perf_counter() - start
            final = estimator.estimate(state.alive_mask)
            if final["F"] is None:
                raise RuntimeError("NOT_CONVERGED: 测试删边后传播未收敛")
            metrics = compute_metrics(graph, state.alive_mask)
            resources = getattr(policy, "training_resources", {})
            row = {"row_type": "graph", "policy_id": name, "method": method, "ablation": ablation,
                   "training_seed": training_seed, "graph_id": graph.graph_id, "group_id": graph.group_id,
                   "split": metadata["split"], "N": graph.N, "M0": graph.M0, "K": K,
                   "evaluation_seed": test_seed, "mc": mc, "F0": initial["F"], "FK": final["F"],
                   "delta_F": initial["F"] - final["F"], "P0": initial["P"], "PK": final["P"],
                   "eta_K": (initial["F"] - final["F"]) / initial["F"] if initial["F"] >= .05 else None,
                   "L1": metrics["L1"], "L2": metrics["L2"], "actions": json.dumps(actions),
                   "inference_seconds": inference_seconds, "evaluation_sir_trajectories": estimator.trajectory_calls,
                   "selection_sir_trajectories": greedy.trajectory_calls if greedy is not None else 0,
                   "training_episodes": resources.get("episodes"), "training_sir_trajectories": resources.get("sir_trajectories"),
                   "validation_sir_trajectories": resources.get("validation_sir_trajectories"),
                   "training_gradient_updates": resources.get("gradient_updates"),
                   "training_optimizer_steps": json.dumps(resources.get("optimizer_steps_by_network", {})),
                   "training_wall_seconds": resources.get("wall_seconds"), "training_cuda_seconds": resources.get("cuda_seconds"),
                   "training_peak_cuda_bytes": resources.get("peak_cuda_bytes"), "science_status": "NOT_EVALUATED"}
            rows.append(row)
    return rows


def _bootstrap(values, rng, resamples):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return None, None  # 一个图不足以估计图级抽样不确定性。
    means = values[rng.integers(len(values), size=(resamples, len(values)))].mean(1)
    return tuple(float(value) for value in np.quantile(means, [.025, .975]))


def summarize(rows, *, seed, resamples=2000):
    """先在每张图内平均训练种子，再按图配对 Bootstrap，不把 MC 轨迹当作图样本。"""
    if not isinstance(resamples, int) or resamples < 1:
        raise ValueError("resamples 必须为正整数")
    groups, random_reference = defaultdict(list), {}
    for row in rows:
        groups[(row["method"], row["ablation"], row["split"])].append(row)
        if row["method"] == "random":
            random_reference[(row["split"], row["graph_id"])] = row["FK"]
    rng = np.random.default_rng(np.random.SeedSequence([seed, 23]))
    summaries = []
    for (method, ablation, split), records in sorted(groups.items()):
        by_graph, by_seed = defaultdict(list), defaultdict(list)
        for row in records:
            by_graph[row["graph_id"]].append(row)
            if row["training_seed"] is not None:
                by_seed[row["training_seed"]].append(row["FK"])
        # 所有种子必须使用完全相同的测试图，防止不完整结果被选择性平均。
        if by_seed and any(len(values) != len(by_graph) for values in by_seed.values()):
            raise ValueError("不同训练种子的测试图集合不一致")
        averages = {graph_id: {metric: float(np.mean([r[metric] for r in values]))
                               for metric in ("FK", "delta_F", "PK", "L1", "L2")}
                    for graph_id, values in by_graph.items()}
        summary = {"row_type": "summary", "method": method, "ablation": ablation, "split": split,
                   "graph_count": len(by_graph), "training_seed_count": len(by_seed),
                   "eta_excluded_count": sum(all(r["eta_K"] is None for r in values) for values in by_graph.values()),
                   "training_seed_FK_std": float(np.std([np.mean(v) for v in by_seed.values()], ddof=1))
                   if len(by_seed) > 1 else None, "science_status": "NOT_EVALUATED"}
        for metric in ("FK", "delta_F", "PK", "L1", "L2"):
            values = [value[metric] for value in averages.values()]
            summary[metric] = float(np.mean(values))
            summary[metric + "_ci_low"], summary[metric + "_ci_high"] = _bootstrap(values, rng, resamples)
        paired = [random_reference[(split, graph_id)] - values["FK"]
                  for graph_id, values in averages.items() if (split, graph_id) in random_reference]
        if paired:
            if len(paired) != len(averages):
                raise ValueError("随机基线与策略的测试图不一致")
            summary["FK_gain_vs_random"] = float(np.mean(paired))
            summary["FK_gain_ci_low"], summary["FK_gain_ci_high"] = _bootstrap(paired, rng, resamples)
        summaries.append(summary)
    return summaries


def save_results(path, rows, *, seed, resamples=2000):
    summary = summarize(rows, seed=seed, resamples=resamples)
    all_rows = list(rows) + summary
    fields = list(dict.fromkeys(key for row in all_rows for key in row))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    return summary


def _smoke_data(config, seed):
    from .generators import generate_graph
    seeds = [int(s) for s in np.random.SeedSequence([seed, 24]).generate_state(5)]
    training = [generate_graph(config, seed=seeds[0])[0]]
    validation = [generate_graph(config, seed=seeds[1])[0]]
    test = [(generate_graph(config, seed=seeds[2])[0], {"split": "test_id"})]
    size_config = deepcopy(config)
    size_config["graph"].update(N=100, M2=200, M3=100)
    test.append((generate_graph(size_config, seed=seeds[3])[0], {"split": "ood_size"}))
    structure_config = deepcopy(config)
    structure_config["graph"]["overlap_strength"] = "strong"
    test.append((generate_graph(structure_config, skeleton="SBM", hyperedge="Overlap", seed=seeds[4])[0],
                 {"split": "ood_structure"}))
    return training, validation, test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="独立测试集上的策略、基线与消融评价")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--checkpoint-dir", default=None, help="保存冒烟训练中由验证集选中的模型")
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--mc", type=int, default=None)
    parser.add_argument("--greedy-mc", type=int, default=None)
    parser.add_argument("--bootstrap", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--training-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--ablations", choices=ABLATIONS, nargs="+", default=["full"])
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = load_config(args.config)
    seed = config["seed"] if args.seed is None else args.seed
    if not args.smoke and (not args.dataset or not args.checkpoint):
        parser.error("实际评价须提供 --dataset 和 --checkpoint；小样本流程请使用 --smoke")
    if args.dataset:
        from .dataset import load_dataset
        samples = load_dataset(args.dataset)
        training = [g for g, m in samples if m["split"] == "train"]
        validation = [g for g, m in samples if m["split"] == "validation"]
        test = [(g, m) for g, m in samples if m["split"] in ("test_id", "ood_size", "ood_structure")]
    else:
        training, validation, test = _smoke_data(config, seed)
    p = config["sir"]
    parameters = SIRParameters(p["beta1"], p["beta2"], p["mu"], p["rho0"], p["max_steps"])
    K = config["rl"]["K"]
    policies = {name: name for name in ("random", "overlap_degree", "sir_greedy")}
    for path in args.checkpoint:
        agent, checkpoint = load_agent(path)
        if checkpoint["smoke"] and not args.smoke:
            parser.error("冒烟模型只能用于 --smoke 评价")
        parameters, K = agent.sir_parameters, checkpoint["K"]
        name = f"{checkpoint['algorithm']}/{agent.config.get('ablation', 'full')}/{agent.seed}"
        if name in policies:
            parser.error("重复 checkpoint")
        policies[name] = agent
    if args.smoke and not args.checkpoint:
        if not training or not validation:
            parser.error("冒烟训练和验证集都必须非空")
        for training_seed in args.training_seeds or [seed, seed + 1]:
            for ablation in args.ablations:
                settings = ablation_config(config, ablation)
                settings["rl"].update(batch_size=2, warmup_transitions=2, validation_interval=1)
                graphs = {graph.graph_id: graph for graph in training}
                for method, agent_class in (("ddqn", DDQNAgent), ("sac", SACAgent)):
                    agent = agent_class(graphs, settings, seed=training_seed)
                    env_seed = int(np.random.SeedSequence([training_seed, 12]).generate_state(1)[0])
                    env = DeletionEnv(graphs, parameters, seed=env_seed, mc=8)
                    resources = train(agent, env, settings, episodes=2, seed=training_seed,
                                      validation_graphs=validation, validation_mc=8)
                    if args.checkpoint_dir:
                        checkpoint_path = Path(args.checkpoint_dir) / f"{method}_{ablation}_seed{training_seed}.pt"
                        save_checkpoint(agent, checkpoint_path, parameters, resources, smoke=True)
                    policies[f"{method}/{ablation}/{training_seed}"] = agent
    mc = (8 if args.smoke else p["test_mc"]) if args.mc is None else args.mc
    greedy_mc = (4 if args.smoke else p["train_mc"]) if args.greedy_mc is None else args.greedy_mc
    bootstrap = (100 if args.smoke else 2000) if args.bootstrap is None else args.bootstrap
    rows = evaluate_policies(test, policies, parameters, K, mc=mc, seed=seed, greedy_mc=greedy_mc)
    summary = save_results(args.output, rows, seed=seed, resamples=bootstrap)
    print(json.dumps({"graphs": len(test), "policies": len(policies), "result_rows": len(rows),
                      "summary_rows": len(summary), "output": str(args.output),
                      "checkpoint_dir": args.checkpoint_dir,
                      "science_status": "NOT_EVALUATED"}, ensure_ascii=False, indent=2))
