"""课程 ABC 与均匀抽图的连续训练；只保存三张结果表和模型 checkpoint。"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import csv
from dataclasses import asdict
import hashlib
from itertools import product
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import torch

from .dataset import _families, _check_splits
from .ddqn import DDQNAgent, validation_score
from .evaluate import evaluate_policies, _bootstrap
from .generators import generate_graph, generate_course_graph
from .hypergraph import compute_metrics
from .rl_env import DeletionEnv, ReplayBuffer, Transition
from .sac import SACAgent
from .sir import SIRParameters
from .utils import load_config


PROFILES = {
    "quick": {"episodes": [60, 90, 150], "train_mc": 32, "val_mc": 128, "test_mc": 512, "seeds": 1},
    "full": {"episodes": [200, 300, 500], "train_mc": 64, "val_mc": 256, "test_mc": 2048, "seeds": 2},
}
SAMPLING = [[.8, .2, 0.], [.3, .5, .2], [.1, .2, .7]]
COUNTS = {"train": [20, 20, 40], "validation": [8, 8, 16], "test_id": [8, 8, 32]}
CURVE_FIELDS = ["method", "algorithm", "sampling", "seed", "episode", "phase", "course_stage",
                "graph_id", "train_reward", "train_F6", "validation_F6", "validation_A_F6",
                "validation_B_F6", "validation_C_F6", "gradient_updates", "replay_size",
                "train_sir_calls", "validation_sir_calls", "wall_seconds", "loss_json"]


def build_pool(config, seed):
    """固定 160 张图，不以传播结果筛选；复用现有家族/同构隔离检查。"""
    combinations = list(product(["ER", "WS", "SBM"],
                                ["Uniform", "Overlap", "Community", "Heterogeneous"]))
    samples = []
    for split_index, (split, counts) in enumerate(COUNTS.items()):
        for course_index, (course, count) in enumerate(zip("ABC", counts)):
            for index in range(count):
                graph_seed = int(np.random.SeedSequence([seed, 100, split_index, course_index, index]).generate_state(1)[0])
                if course == "C":
                    # 各划分每种组合相差至多一张；跨划分轮换余数，避免额外图总偏向同一机制。
                    skeleton, hyperedge = combinations[(index + 4 * split_index) % len(combinations)]
                    graph, metadata = generate_graph(config, skeleton, hyperedge, graph_seed)
                else:
                    graph, metadata = generate_course_graph(course, seed=graph_seed,
                                                              cross_binary=[4, 6, 10][index % 3])
                metadata.update(course_stage=course, split=split)
                metrics = compute_metrics(graph, community_labels=metadata["community_labels"])
                metadata["metrics"] = {k: v.tolist() if isinstance(v, np.ndarray) else v
                                       for k, v in metrics.items()}
                samples.append((graph, metadata))
    families = _families(samples)
    _check_splits(samples, {g.graph_id: m["split"] for g, m in samples}, families)
    for graph, metadata in samples:
        metadata["family_id"] = families[graph.graph_id]
    print("POOL", dict(Counter((m["split"], m["course_stage"]) for _, m in samples)), flush=True)
    print("C_GENERATORS", dict(Counter((m["split"], m["generator_type"]) for _, m in samples
                                       if m["course_stage"] == "C")), flush=True)
    return samples


def pool_records(samples):
    return [{"graph_id": g.graph_id, "group_id": g.group_id, "N": g.N,
             "edge_index_2": g.edge_index_2.tolist(), "hyperedge_nodes_3": g.hyperedge_nodes_3.tolist(),
             "metadata": m} for g, m in samples]


def save_training(agent, replay, rng, env, samples, history, best_state, best_score,
                  best_episode, resources, path, experiment):
    """同一 checkpoint 同时保留验证选中的推断模型和最后的训练状态。"""
    if isinstance(agent, DDQNAgent):
        training = {"online": agent.online.state_dict(), "target": agent.target.state_dict(),
                    "optimizer": agent.optimizer.state_dict()}
    else:
        training = {"actor": agent.actor.state_dict(), "critics": agent.critics.state_dict(),
                    "targets": agent.targets.state_dict(), "log_alpha": agent.log_alpha.detach(),
                    "actor_optimizer": agent.actor_optimizer.state_dict(),
                    "critic_optimizers": [o.state_dict() for o in agent.critic_optimizers],
                    "alpha_optimizer": agent.alpha_optimizer.state_dict()}
    transitions = [dict(graph_id=t.graph_id, mask_before=t.mask_before.tolist(),
                        action_edge_id=int(t.action_edge_id), reward=t.reward, mask_after=t.mask_after.tolist(),
                        done=bool(t.done), K=t.K, episode_id=t.episode_id, sir_config_id=t.sir_config_id)
                   for t in replay._items]
    training.update(environment_steps=agent.environment_steps, gradient_updates=agent.gradient_updates,
                    episode_id=env.episode_id, replay=transitions, replay_position=replay.position,
                    replay_rng=replay.rng.bit_generator.state, agent_rng=agent.rng.bit_generator.state,
                    sampling_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state())
    torch.save({"algorithm": "ddqn" if isinstance(agent, DDQNAgent) else "sac", "config": agent.config,
                "seed": agent.seed, "model": best_state, "sir_parameters": asdict(env.parameters),
                "K": 6, "best_validation_FK": best_score, "best_episode": best_episode,
                "provenance": agent.provenance, "resources": resources, "smoke": False,
                "experiment": experiment, "dataset": pool_records(samples),
                "training_state": training, "learning_curve": history}, path)


def train_one(method, sampling, seed, config, experiment, samples, output):
    torch.set_num_threads(1)
    profile = experiment["settings"]
    training = [(g, m) for g, m in samples if m["split"] == "train"]
    validation = [(g, m) for g, m in samples if m["split"] == "validation"]
    graphs = {g.graph_id: g for g, _ in training}
    pools = [[g for g, m in training if m["course_stage"] == c] for c in "ABC"]
    agent = {"DDQN": DDQNAgent, "SAC": SACAgent}[method](graphs, config, seed=seed, device=experiment["device"])
    provenance = training + validation
    agent.provenance = {"graph_ids": [g.graph_id for g, _ in provenance],
                        "group_ids": sorted({m["family_id"] for _, m in provenance} | {g.group_id for g, _ in provenance}),
                        "skeletons": [{"graph_id": g.graph_id, "group_id": g.group_id,
                                       "N": g.N, "edge_index_2": g.edge_index_2.tolist()} for g, _ in provenance]}
    s = config["sir"]
    parameters = SIRParameters(s["beta1"], s["beta2"], s["mu"], s["rho0"], s["max_steps"])
    agent.sir_parameters = parameters
    env_seed = int(np.random.SeedSequence([seed, 12]).generate_state(1)[0])
    env = DeletionEnv(graphs, parameters, seed=env_seed, mc=profile["train_mc"])
    # 仅构造一次；三个阶段只改变图采样概率，持续保留优化器、Replay 和步数。
    replay = ReplayBuffer(config["rl"]["capacity"], seed)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 10]))
    method_name = f"{sampling}-{method}"
    ends = np.cumsum(profile["episodes"])
    best_state, best_score, best_episode = None, None, None
    history, val_calls = [], 0
    started = time.perf_counter()
    p = config["rl"]
    path = Path(output) / f"{sampling.lower()}_{method.lower()}_seed{seed}.pt"
    course_by_id = {g.graph_id: m["course_stage"] for g, m in training}

    def validate():
        nonlocal val_calls
        scores = {}
        for index, course in enumerate("ABC"):
            group = [g for g, m in validation if m["course_stage"] == course]
            validation_seed = int(np.random.SeedSequence([experiment["data_seed"], 110, index]).generate_state(1)[0])
            score, calls = validation_score(agent, group, parameters, 6, mc=profile["val_mc"], seed=validation_seed)
            scores[f"validation_{course}_F6"] = score
            val_calls += calls
        scores["validation_F6"] = sum(scores[f"validation_{c}_F6"] * n for c, n in zip("ABC", COUNTS["validation"])) / len(validation)
        return scores

    initial = {"method": method_name, "algorithm": method, "sampling": sampling, "seed": seed,
               "episode": 0, "phase": 0, "gradient_updates": 0, "replay_size": 0, "train_sir_calls": 0}
    initial.update(validate(), validation_sir_calls=val_calls, wall_seconds=time.perf_counter()-started)
    history.append(initial)
    print(f"{method_name} seed={seed}: initial validation F6={initial['validation_F6']:.6f}", flush=True)
    for episode in range(1, int(ends[-1])+1):
        phase = int(np.searchsorted(ends, episode, side="left"))
        if sampling == "Curriculum":
            pool = pools[int(rng.choice(3, p=SAMPLING[phase]))]
            graph = pool[int(rng.integers(len(pool)))]
        else:
            graph = training[int(rng.integers(len(training)))][0]
        state = env.reset(graph.graph_id, 6)
        total, loss = 0., None
        while not state.done:
            action = (int(rng.choice(np.flatnonzero(state.alive_mask))) if len(replay) < p["warmup_transitions"]
                      else agent.act(state, explore=True))
            next_state, reward, done, _ = env.step(action)
            replay.add(Transition(state.graph_id, state.alive_mask, action, reward, next_state.alive_mask,
                                  done, 6, env.episode_id, env.sir_config_id))
            agent.environment_steps += 1
            if len(replay) >= max(p["warmup_transitions"], p["batch_size"]):
                loss = agent.update(replay.sample(p["batch_size"]))
            total += reward
            state = next_state
        row = {"method": method_name, "algorithm": method, "sampling": sampling, "seed": seed,
               "episode": episode, "phase": phase+1, "course_stage": course_by_id[graph.graph_id],
               "graph_id": graph.graph_id, "train_reward": total,
               "train_F6": env.monte_carlo.estimate(state.alive_mask)["F"],
               "gradient_updates": agent.gradient_updates, "replay_size": len(replay),
               "train_sir_calls": env.sir_trajectories, "loss_json": json.dumps(loss)}
        validation_due = episode % p["validation_interval"] == 0 or episode == ends[-1]
        if validation_due:
            row.update(validate())
            if best_score is None or row["validation_F6"] < best_score:
                best_score, best_episode = row["validation_F6"], episode
                best_state = agent.inference_state()
            print(f"{method_name} ep={episode}: validation F6={row['validation_F6']:.6f}, best={best_score:.6f} at {best_episode}", flush=True)
        row.update(validation_sir_calls=val_calls, wall_seconds=time.perf_counter()-started)
        history.append(row)
        if episode % 10 == 0 or episode in ends:
            recent = [r["train_reward"] for r in history[-10:] if "train_reward" in r]
            print(f"{method_name} ep={episode}/{ends[-1]} phase={phase+1} reward10={np.mean(recent):.6f} "
                  f"updates={agent.gradient_updates} replay={len(replay)} seconds={row['wall_seconds']:.1f}", flush=True)
        if validation_due:
            resources = {"episodes": episode, "gradient_updates": agent.gradient_updates,
                         "sir_trajectories": env.sir_trajectories, "validation_sir_trajectories": val_calls,
                         "best_validation_FK": best_score, "best_episode": best_episode,
                         "wall_seconds": time.perf_counter()-started,
                         "optimizer_steps_by_network": {key: getattr(agent, key) for key in
                             ("critic_updates", "actor_updates", "temperature_updates") if hasattr(agent, key)}}
            save_training(agent, replay, rng, env, samples, history, best_state, best_score,
                          best_episode, resources, path, experiment)
    agent.load_inference_state(best_state)
    agent.training_resources = resources
    test = [(g, m) for g, m in samples if m["split"] == "test_id"]
    results = evaluate_policies(test, {method_name: agent}, parameters, 6, mc=profile["test_mc"], seed=experiment["data_seed"])
    print(f"DONE {method_name} seed={seed}: {len(results)} test graphs; {time.perf_counter()-started:.1f}s", flush=True)
    return history, results


def result_rows(rows, samples):
    metadata = {g.graph_id: m for g, m in samples}
    for row in rows:
        m = metadata[row["graph_id"]]
        row.update(course_stage=m["course_stage"], generator_type=m["generator_type"], graph_seed=m["seed"],
                   F6=row.pop("FK"), delta_F6=row.pop("delta_F"), science_status="EXPLORATORY")
        row["method"] = {"random": "Random", "overlap_degree": "Overlap-Degree"}.get(row["policy_id"], row["policy_id"])
    return rows


def summaries(rows, seed):
    """每图先跨训练种子求均值，按图配对比较课程/均匀与随机基线。"""
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[row["course_stage"], row["method"]][row["graph_id"]].append(row)
    means = {key: {gid: {metric: float(np.mean([r[metric] for r in values]))
                        for metric in ("F0", "F6", "delta_F6")} for gid, values in graphs.items()}
             for key, graphs in groups.items()}
    rng = np.random.default_rng(np.random.SeedSequence([seed, 111]))
    output = []
    for (course, method), graphs in sorted(means.items()):
        row = {"course_stage": course, "method": method, "graph_count": len(graphs),
               "training_seed_count": len({r["training_seed"] for values in groups[course, method].values()
                                            for r in values if r["training_seed"] is not None})}
        for metric in ("F0", "F6", "delta_F6"):
            values = [g[metric] for g in graphs.values()]
            row[metric] = float(np.mean(values))
            row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = _bootstrap(values, rng, 2000)
        for reference, label in [("Random", "gain_vs_random"),
                                 (method.replace("Curriculum-", "Uniform-"), "gain_vs_uniform")]:
            if label == "gain_vs_uniform" and not method.startswith("Curriculum-"):
                continue
            values = [means[course, reference][gid]["F6"]-g["F6"] for gid, g in graphs.items()]
            row[label] = float(np.mean(values))
            row[label+"_ci_low"], row[label+"_ci_high"] = _bootstrap(values, rng, 2000)
        records = [r for values in groups[course, method].values() for r in values]
        row["test_sir_calls"] = sum(r["evaluation_sir_trajectories"] for r in records)
        row["science_status"] = "EXPLORATORY"
        output.append(row)
    return output


def write_csv(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="quick")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4, help="独立实验进程数；各进程单 CPU 线程")
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = load_config()
    config["graph"].update(N=30, M2=60, M3=30)
    config["rl"].update(K=6, validation_interval=100)
    profile = deepcopy(PROFILES[args.profile])
    config["sir"].update(train_mc=profile["train_mc"], val_mc=profile["val_mc"], test_mc=profile["test_mc"])
    seed = config["seed"] if args.seed is None else args.seed
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable: using CPU.", flush=True)
        device = "cpu"
    if args.workers < 1:
        parser.error("workers 必须至少为 1")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        parser.error("输出目录已非空，请指定新目录以保留原始结果")
    output.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        source_hash.update(path.name.encode()+path.read_bytes())
    experiment = {"profile": args.profile, "settings": profile, "all_profiles": PROFILES,
                  "sampling_probabilities": SAMPLING, "pool_counts": COUNTS, "data_seed": seed,
                  "device": device, "source_sha256": source_hash.hexdigest(), "workers": args.workers,
                  "model_selection": "lowest graph-mean validation F6 at every 100 episodes",
                  "torch_version": str(torch.__version__), "numpy_version": np.__version__}
    started = time.perf_counter()
    samples = build_pool(config, seed)
    jobs = [(method, sampling, seed+i) for i in range(profile["seeds"])
            for method in ("DDQN", "SAC") for sampling in ("Curriculum", "Uniform")]
    histories, results = [], []
    s = config["sir"]
    parameters = SIRParameters(s["beta1"], s["beta2"], s["mu"], s["rho0"], s["max_steps"])
    print("EXPERIMENT", json.dumps(experiment), flush=True)
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs)), mp_context=mp.get_context("spawn")) as executor:
        futures = [executor.submit(train_one, method, sampling, training_seed, config, experiment, samples, str(output))
                   for method, sampling, training_seed in jobs]
        test = [(g, m) for g, m in samples if m["split"] == "test_id"]
        results.extend(result_rows(evaluate_policies(test, {"random": "random", "overlap_degree": "overlap_degree"},
                                                   parameters, 6, mc=profile["test_mc"], seed=seed), samples))
        write_csv(output / "results.csv", results)
        print("Random and Overlap-Degree finished on all 48 fixed test graphs.", flush=True)
        for future in as_completed(futures):
            history, rows = future.result()
            histories.extend(history)
            results.extend(result_rows(rows, samples))
            write_csv(output / "learning_curve.csv", sorted(histories, key=lambda r:(r["method"],r["seed"],r["episode"])), CURVE_FIELDS)
            write_csv(output / "results.csv", results)
    summary = summaries(results, seed)
    write_csv(output / "summary.csv", summary)
    final_curves = [r for r in histories if r["episode"] == sum(profile["episodes"])]
    print("COMPLETE", json.dumps({"wall_seconds": time.perf_counter()-started,
          "models": len(jobs), "episodes": sum(r["episode"] for r in final_curves),
          "training_sir_calls": sum(r["train_sir_calls"] for r in final_curves),
          "validation_sir_calls": sum(r["validation_sir_calls"] for r in final_curves),
          "test_sir_calls": sum(r["evaluation_sir_trajectories"] for r in results),
          "output": str(output), "summary": summary}), flush=True)


if __name__ == "__main__":
    main()
