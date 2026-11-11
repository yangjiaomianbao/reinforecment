"""完整 HGNN + Double DQN；训练入口默认要求显式选择冒烟或提供数据集。"""

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import time

import networkx as nx
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .hgnn import HGNN, ABLATIONS, ablation_config
from .rl_env import State, Transition, DeletionEnv, ReplayBuffer, action_features
from .sir import SIRParameters, SIRMonteCarlo
from .utils import load_config, set_seed


class ActionNetwork(nn.Module):
    """共享逐动作评分头；每个实例拥有独立的完整编码器。"""
    def __init__(self, config):
        super().__init__()
        self.encoder = HGNN.from_config(config)
        dim = self.encoder.hidden_dim
        self.head = nn.Sequential(nn.Linear(2 * dim + 1, dim), nn.ReLU(), nn.Linear(dim, 1))

    def forward(self, graph, state):
        features, valid = action_features(self.encoder, graph, state)
        return self.head(features).squeeze(-1).masked_fill(~valid, -torch.inf)


@torch.no_grad()
def soft_update(target, online, tau):
    if not 0 < tau <= 1:
        raise ValueError("target_tau 必须在 (0,1] 内")
    for target_parameter, parameter in zip(target.parameters(), online.parameters()):
        target_parameter.lerp_(parameter, tau)


class DDQNAgent:
    def __init__(self, graphs, config, *, seed=0, device="cpu"):
        if config["rl"]["gamma"] != 1:
            raise ValueError("主实验要求 gamma=1")
        self.graphs, self.config, self.seed = dict(graphs), deepcopy(config), seed
        self.rng = np.random.default_rng(set_seed(seed, "rl"))
        self.online = ActionNetwork(config).to(device)
        self.target = deepcopy(self.online).requires_grad_(False).eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config["rl"]["learning_rate"])
        self.environment_steps = self.gradient_updates = 0

    def act(self, state, *, graph=None, explore=False):
        if state.done:
            raise ValueError("终止状态不能选择动作")
        graph = self.graphs[state.graph_id] if graph is None else graph
        p = self.config["rl"]
        if p["epsilon_decay_steps"] <= 0:
            raise ValueError("epsilon_decay_steps 必须为正数")
        fraction = min(1., self.environment_steps / p["epsilon_decay_steps"])
        epsilon = p["epsilon_start"] + fraction * (p["epsilon_end"] - p["epsilon_start"])
        if explore and self.rng.random() < epsilon:
            return int(self.rng.choice(np.flatnonzero(state.alive_mask)))
        with torch.no_grad():
            return int(self.online(graph, state).argmax().item())

    @torch.no_grad()
    def td_targets(self, transitions):
        values = []
        for transition in transitions:
            value = next(self.online.parameters()).new_tensor(transition.reward)
            # 终止时直接用 r，不能先对空动作集求 argmax。
            if not transition.done:
                graph, state = self.graphs[transition.graph_id], transition.next_state
                action = self.online(graph, state).argmax()
                value = value + self.target(graph, state)[action]
            values.append(value)
        return torch.stack(values)

    def update(self, transitions, *, update_target=True):
        targets = self.td_targets(transitions)
        predictions = torch.stack([self.online(self.graphs[t.graph_id], t.state)[t.action_edge_id]
                                   for t in transitions])
        loss = F.smooth_l1_loss(predictions, targets)
        if not torch.isfinite(loss):
            raise RuntimeError("DDQN TD loss 非有限值")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.gradient_updates += 1
        if update_target:
            soft_update(self.target, self.online, self.config["rl"]["target_tau"])
        return {"loss": float(loss.detach())}

    def inference_state(self):
        return deepcopy(self.online.state_dict())

    def load_inference_state(self, state):
        self.online.load_state_dict(state)
        self.target.load_state_dict(state)


def validation_score(agent, graphs, parameters, K, *, mc, seed):
    """只用固定独立验证随机组比较模型；不使用训练 SIR 随机组。"""
    scores, calls = [], 0
    for graph in graphs:
        state = State(graph.graph_id, np.ones(graph.M0, dtype=bool), K, K)
        while not state.done:
            action = agent.act(state, graph=graph, explore=False)
            mask = state.alive_mask.copy()
            mask[action] = False
            state = State(graph.graph_id, mask, state.remaining_budget - 1, K)
        graph_seed = int(np.random.SeedSequence([seed, 9, len(scores)]).generate_state(1)[0])
        estimator = SIRMonteCarlo(graph, parameters, seed=graph_seed, mc=mc)
        estimate = estimator.estimate(state.alive_mask)
        if estimate["F"] is None:
            raise RuntimeError("NOT_CONVERGED: 验证 SIR 未达到吸收态")
        scores.append(estimate["F"])
        calls += estimator.trajectory_calls
    if not scores:
        raise ValueError("验证集不能为空")
    return float(np.mean(scores)), calls


def ensure_unseen(graph, provenance):
    """同时检查 ID、祖先组与重编号后的二元骨架，防止来源标签变化绕过隔离。"""
    if graph.graph_id in provenance.get("graph_ids", []) or graph.group_id in provenance.get("group_ids", []):
        raise ValueError("SPLIT_LEAKAGE: 留出图与训练/选模图共享 ID 或祖先组")
    network = nx.Graph()
    network.add_nodes_from(range(graph.N))
    network.add_edges_from(graph.edge_index_2)
    degrees = sorted(dict(network.degree()).values())
    for record in provenance.get("skeletons", []):
        if record["N"] != graph.N or len(record["edge_index_2"]) != graph.M2:
            continue
        other = nx.Graph()
        other.add_nodes_from(range(record["N"]))
        other.add_edges_from(record["edge_index_2"])
        if degrees == sorted(dict(other.degree()).values()) and nx.is_isomorphic(network, other):
            raise ValueError("SPLIT_LEAKAGE: 留出图的二元骨架与训练/选模图同构")


def train(agent, env, config, *, episodes, seed, validation_graphs=None, validation_mc=None,
          sir_budget=None):
    """少量或显式指定规模的离策略训练；返回实际资源和验证选择结果。"""
    p = config["rl"]
    if not isinstance(episodes, int) or episodes < 1 or p["validation_interval"] < 1:
        raise ValueError("episodes 和 validation_interval 必须为正整数")
    if max(p["batch_size"], p["warmup_transitions"]) > p["capacity"]:
        raise ValueError("Replay 容量不能小于 batch_size 或 warmup_transitions")
    graphs = list(env.graphs.values())
    if not graphs or any(graph.M0 < p["K"] for graph in graphs):
        raise ValueError("训练图不能为空且必须有足够超边满足 K")
    seen = {record["graph_id"]: record for record in getattr(agent, "provenance", {}).get("skeletons", [])}
    for graph in graphs + list(validation_graphs or []):
        seen[graph.graph_id] = {"graph_id": graph.graph_id, "group_id": graph.group_id,
                                "N": graph.N, "edge_index_2": graph.edge_index_2.tolist()}
    agent.provenance = {"graph_ids": sorted(seen),
                        "group_ids": sorted({record["group_id"] for record in seen.values()}),
                        "skeletons": list(seen.values())}
    agent.sir_parameters = env.parameters
    training_provenance = {"graph_ids": [g.graph_id for g in graphs],
                           "group_ids": [g.group_id for g in graphs],
                           "skeletons": [seen[g.graph_id] for g in graphs]}
    for graph in validation_graphs or []:
        ensure_unseen(graph, training_provenance)
    replay = ReplayBuffer(p["capacity"], seed)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 10]))
    history, best_state, best_score = [], None, None
    validation_calls = 0
    initial_calls, initial_updates = env.sir_trajectories, agent.gradient_updates
    network_updates = {name: getattr(agent, name) for name in
                       ("critic_updates", "actor_updates", "temperature_updates") if hasattr(agent, name)}
    device = next((agent.online if hasattr(agent, "online") else agent.actor).parameters()).device
    cuda_start = cuda_end = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        cuda_start, cuda_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        cuda_start.record(torch.cuda.current_stream(device))
    start = time.perf_counter()
    for episode in range(episodes):
        expected_calls = (p["K"] + 1) * env.mc
        if sir_budget is not None and env.sir_trajectories - initial_calls + expected_calls > sir_budget:
            break  # 不为预算上限截断一个本应恰删 K 条边的 Episode。
        graph = graphs[int(rng.integers(len(graphs)))]
        state = env.reset(graph.graph_id, p["K"])
        total, last_loss = 0., None
        while not state.done:
            action = (int(rng.choice(np.flatnonzero(state.alive_mask)))
                      if len(replay) < p["warmup_transitions"] else agent.act(state, explore=True))
            next_state, reward, done, info = env.step(action)
            replay.add(Transition(state.graph_id, state.alive_mask, action, reward,
                                  next_state.alive_mask, done, state.K, env.episode_id, env.sir_config_id))
            agent.environment_steps += 1
            if len(replay) >= max(p["warmup_transitions"], p["batch_size"]):
                last_loss = agent.update(replay.sample(p["batch_size"]))
            state, total = next_state, total + reward
        row = {"episode": episode + 1, "return": total,
               "FK": env.monte_carlo.estimate(state.alive_mask)["F"], "last_update": last_loss}
        if validation_graphs and ((episode + 1) % p["validation_interval"] == 0 or episode + 1 == episodes):
            score, calls = validation_score(agent, validation_graphs, env.parameters, p["K"],
                                             mc=validation_mc or config["sir"]["val_mc"], seed=seed)
            validation_calls += calls
            row["validation_FK"] = score
            if best_score is None or score < best_score:
                best_score, best_state = score, agent.inference_state()
        history.append(row)
    if not history:
        raise ValueError("SIR 预算不足以运行一个完整 Episode")
    if validation_graphs and best_state is None:
        best_score, calls = validation_score(agent, validation_graphs, env.parameters, p["K"],
                                              mc=validation_mc or config["sir"]["val_mc"], seed=seed)
        validation_calls += calls
        best_state = agent.inference_state()
    if best_state is not None:
        agent.load_inference_state(best_state)
    cuda_seconds = None
    if cuda_end is not None:
        cuda_end.record(torch.cuda.current_stream(device))
        torch.cuda.synchronize(device)
        cuda_seconds = cuda_start.elapsed_time(cuda_end) / 1000
    result = {"history": history, "best_validation_FK": best_score,
            "optimizer_steps_by_network": {name: getattr(agent, name) - count for name, count in network_updates.items()},
            "episodes": len(history), "gradient_updates": agent.gradient_updates - initial_updates,
            "sir_trajectories": env.sir_trajectories - initial_calls,
            "validation_sir_trajectories": validation_calls,
            "wall_seconds": time.perf_counter() - start, "cuda_seconds": cuda_seconds,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
            "science_status": "NOT_EVALUATED"}
    agent.training_resources = {key: value for key, value in result.items() if key != "history"}
    return result


def save_checkpoint(agent, path, parameters, resources, *, smoke=False):
    """保存验证选中的推断模型及数据来源；不将其声明为可完整恢复的训练快照。"""
    checkpoint = {"algorithm": "ddqn" if hasattr(agent, "online") else "sac",
                  "config": agent.config, "seed": agent.seed, "model": agent.inference_state(),
                  "sir_parameters": asdict(parameters), "K": agent.config["rl"]["K"],
                  "best_validation_FK": resources.get("best_validation_FK"),
                  "provenance": getattr(agent, "provenance", {}), "smoke": smoke,
                  "resources": {key: value for key, value in resources.items() if key != "history"}}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as file:
        torch.save(checkpoint, file)


def training_main(agent_class, algorithm):
    parser = argparse.ArgumentParser(description=f"{algorithm} HGNN 删边策略训练")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--mc", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ablation", choices=ABLATIONS, default="full")
    args = parser.parse_args()
    config = ablation_config(load_config(args.config), args.ablation)
    seed = config["seed"] if args.seed is None else args.seed
    if not args.smoke and (args.dataset is None or args.episodes is None):
        parser.error("请用 --smoke 运行小样本检查；实际训练须显式指定 --dataset 和 --episodes")
    if args.smoke:
        torch.set_num_threads(1)
        config["rl"].update(batch_size=2, warmup_transitions=2, validation_interval=1)
    episodes = (3 if args.smoke else None) if args.episodes is None else args.episodes
    mc = (8 if args.smoke else config["sir"]["train_mc"]) if args.mc is None else args.mc
    if args.dataset:
        from .dataset import load_dataset
        records = load_dataset(args.dataset)
        training = [g for g, metadata in records if metadata["split"] == "train"]
        validation = [g for g, metadata in records if metadata["split"] == "validation"]
    else:
        from .generators import generate_graph
        seeds = np.random.SeedSequence([seed, 11]).generate_state(2)
        training = [generate_graph(config, seed=int(seeds[0]))[0]]
        validation = [generate_graph(config, seed=int(seeds[1]))[0]]
    if not training or not validation:
        parser.error("训练与验证集都必须非空")
    p = config["sir"]
    parameters = SIRParameters(p["beta1"], p["beta2"], p["mu"], p["rho0"], p["max_steps"])
    graphs = {graph.graph_id: graph for graph in training}
    agent = agent_class(graphs, config, seed=seed, device=args.device)
    env_seed = int(np.random.SeedSequence([seed, 12]).generate_state(1)[0])
    env = DeletionEnv(graphs, parameters, seed=env_seed, mc=mc, gamma=config["rl"]["gamma"])
    result = train(agent, env, config, episodes=episodes, seed=seed, validation_graphs=validation,
                   validation_mc=mc if args.smoke else p["val_mc"])
    if args.output:
        if args.dataset:
            family_ids = {metadata["family_id"] for _, metadata in records if metadata["split"] in ("train", "validation")}
            agent.provenance["group_ids"] = sorted(set(agent.provenance["group_ids"]) | family_ids)
        save_checkpoint(agent, args.output, parameters, result, smoke=args.smoke)
    print(json.dumps({"algorithm": algorithm, "ablation": args.ablation, "seed": seed, **result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    training_main(DDQNAgent, "ddqn")
