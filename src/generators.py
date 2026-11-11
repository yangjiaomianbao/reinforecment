"""综合混合超图单图生成，不执行批量数据集生成或参数校准。"""

import argparse
from copy import deepcopy
import hashlib
from itertools import combinations
import json
from math import comb

import networkx as nx
import numpy as np

from .hypergraph import Graph, compute_metrics
from .utils import load_config, set_seed


def _probabilities(values):
    probabilities = np.asarray(values, dtype=float)
    if (probabilities.shape != (3,) or not np.all(np.isfinite(probabilities))
            or np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1)):
        raise ValueError("三项提案概率必须非负、有限且和为 1")
    return probabilities / probabilities.sum()


def _skeleton(N, M2, kind, rng, labels, parameters):
    if kind == "WS":
        k = parameters["ws_k"]
        probability = parameters["ws_rewire_probability"]
        if not isinstance(k, int) or not 0 < k < N or k % 2:
            raise ValueError("WS 要求 k 为小于 N 的正偶数")
        if M2 != N * k // 2:
            raise ValueError(f"WS 的 M2 必须为 N*k/2={N*k//2}，不能静默增删边")
        if not 0 <= probability <= 1:
            raise ValueError("WS 重连概率必须在 [0, 1] 内")
        graph = nx.watts_strogatz_graph(N, k, probability, seed=int(rng.integers(2**32)))
        return np.asarray(list(graph.edges()), dtype=np.int32).reshape(-1, 2)

    pairs = np.asarray(list(combinations(range(N), 2)), dtype=np.int32).reshape(-1, 2)
    probabilities = None
    if kind == "SBM":
        ratio = parameters["sbm_within_between_ratio"]
        if not np.isfinite(ratio) or ratio <= 0:
            raise ValueError("SBM 社区内外采样权重比必须为有限正数")
        weights = np.where(labels[pairs[:, 0]] == labels[pairs[:, 1]], ratio, 1.0)
        probabilities = weights / weights.sum() if len(weights) else None
    elif kind != "ER":
        raise ValueError(f"未知二元骨架: {kind}")
    # ER 均匀、SBM 按社区权重无放回抽样，二者都严格保持 M2。
    return pairs[rng.choice(len(pairs), size=M2, replace=False, p=probabilities)]


def _community_proposal(rng, communities, probabilities):
    pattern = rng.choice(3, p=probabilities)
    if pattern == 0:  # 3:0
        eligible = [i for i, nodes in enumerate(communities) if len(nodes) >= 3]
        if eligible:
            return rng.choice(communities[rng.choice(eligible)], 3, replace=False)
    elif pattern == 1:  # 2:1，有序选择双节点社区和单节点社区。
        eligible = [(i, j) for i in range(3) for j in range(3)
                    if i != j and len(communities[i]) >= 2 and len(communities[j]) >= 1]
        if eligible:
            i, j = eligible[rng.integers(len(eligible))]
            return np.concatenate([rng.choice(communities[i], 2, replace=False),
                                   rng.choice(communities[j], 1)])
    elif all(len(nodes) for nodes in communities):  # 1:1:1
        return np.asarray([rng.choice(nodes) for nodes in communities])
    return None


def _hyperedges(N, M3, kind, rng, labels, weights, parameters):
    if kind not in ("Uniform", "Overlap", "Community", "Heterogeneous"):
        raise ValueError(f"未知三元生成机制: {kind}")
    if kind == "Overlap":
        probabilities = _probabilities(
            parameters["overlap_probabilities"][parameters["overlap_strength"]]
        )
    elif kind == "Community":
        probabilities = _probabilities(parameters["community_pattern_probabilities"])
    communities = [np.flatnonzero(labels == i) for i in range(3)]
    max_attempts = parameters["max_attempts_per_edge"]
    if not isinstance(max_attempts, int) or max_attempts < 1:
        raise ValueError("max_attempts_per_edge 必须为正整数")
    accepted, seen = [], set()
    for _ in range(max_attempts * M3):
        if len(accepted) == M3:
            break
        if kind == "Community":
            candidate = _community_proposal(rng, communities, probabilities)
        elif kind == "Overlap" and accepted:
            operation = rng.choice(3, p=probabilities)
            if operation:
                anchor = np.asarray(accepted[rng.integers(len(accepted))])
                outside = np.setdiff1d(np.arange(N), anchor)
                if len(outside) < 3 - operation:
                    continue
                candidate = np.concatenate([rng.choice(anchor, operation, replace=False),
                                           rng.choice(outside, 3 - operation, replace=False)])
            else:
                candidate = rng.choice(N, 3, replace=False)
        else:
            candidate = rng.choice(N, 3, replace=False,
                                   p=weights if kind == "Heterogeneous" else None)
        if candidate is not None:
            edge = tuple(sorted(map(int, candidate)))
            if edge not in seen:
                accepted.append(edge)
                seen.add(edge)
    if len(accepted) != M3:
        raise RuntimeError(f"GENERATION_FAILED: {kind} 在 {max_attempts*M3} 次提案内"
                           f"仅生成 {len(accepted)}/{M3} 条三元超边")
    return np.asarray(accepted, dtype=np.int32).reshape(-1, 3)


def generate_graph(config=None, skeleton="ER", hyperedge="Uniform", seed=None,
                   graph_id=None, group_id=None):
    """返回 (Graph, metadata)，同一 seed/config 下二元骨架不受三元机制影响。

    metadata 保留参数、社区标签和实际节点权重；这些候选设定尚未校准。
    Overlap 的概率控制提案操作，不保证所得图的 C2 排序或目标范围。
    WS 保持自身 N*k/2 边数；ER/SBM 使用精确 M2 的无放回抽样。
    """
    config = load_config() if config is None else config
    parameters = config["graph"]
    seed = config["seed"] if seed is None else seed
    N, M2, M3 = (parameters[key] for key in ("N", "M2", "M3"))
    if any(not isinstance(value, int) or isinstance(value, bool) for value in (N, M2, M3)):
        raise ValueError("N、M2、M3 必须为整数")
    if N < 1 or not 0 <= M2 <= comb(N, 2) or not 0 <= M3 <= comb(N, 3):
        raise ValueError("节点数或边数超出简单混合超图的可行范围")

    # 分离标签、骨架和三元层随机流，三元生成路径不会改变二元骨架。
    child_seeds = np.random.SeedSequence(set_seed(seed, "generator")).spawn(3)
    label_rng, binary_rng, triple_rng = map(np.random.default_rng, child_seeds)
    labels = np.asarray(parameters.get("community_labels", label_rng.permutation(np.arange(N) % 3)))
    if labels.shape != (N,) or not np.issubdtype(labels.dtype, np.integer) or not np.all(np.isin(labels, [0, 1, 2])):
        raise ValueError("community_labels 必须为长度 N、取值 0/1/2 的整数数组")
    binary_edges = _skeleton(N, M2, skeleton, binary_rng, labels, parameters)

    weights = None
    if hyperedge == "Heterogeneous":
        if "node_weights" in parameters:
            weights = np.asarray(parameters["node_weights"], dtype=float)
        else:
            sigma = parameters["heterogeneous_log_weight_sigma"]
            if not np.isfinite(sigma) or sigma < 0:
                raise ValueError("对数节点权重的标准差必须为有限非负数")
            weights = triple_rng.lognormal(sigma=sigma, size=N)
        if (weights.shape != (N,) or not np.all(np.isfinite(weights))
                or np.any(weights <= 0) or not np.isfinite(weights.sum())):
            raise ValueError("node_weights 必须为长度 N 的有限正数数组且总和有限")
        weights = weights / weights.sum()
    triples = _hyperedges(N, M3, hyperedge, triple_rng, labels, weights, parameters)
    graph = Graph("pending", "pending", N, binary_edges, triples)
    binary_digest = hashlib.sha256(np.asarray([N], dtype="<i8").tobytes()
                                   + graph.edge_index_2.astype("<i4").tobytes())
    graph_digest = binary_digest.copy()
    graph_digest.update(graph.hyperedge_nodes_3.astype("<i4").tobytes())
    graph = Graph(graph_id if graph_id is not None else "graph_" + graph_digest.hexdigest()[:16],
                  group_id if group_id is not None else "skeleton_" + binary_digest.hexdigest()[:16],
                  N, graph.edge_index_2, graph.hyperedge_nodes_3)
    metadata = {
        "seed": int(seed),
        "generator_type": f"{skeleton}/{hyperedge}",
        "parameters": deepcopy(parameters),
        "community_labels": labels.tolist(),
        "node_weights": weights.tolist() if weights is not None else None,
    }
    return graph, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成一张候选混合超图并输出结构统计")
    parser.add_argument("--config", default=None)
    parser.add_argument("--skeleton", choices=["ER", "WS", "SBM"], default="ER")
    parser.add_argument("--hyperedge", choices=["Uniform", "Overlap", "Community", "Heterogeneous"], default="Uniform")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    graph, metadata = generate_graph(load_config(args.config), args.skeleton, args.hyperedge, args.seed)
    metrics = compute_metrics(graph)
    summary = {key: value for key, value in metrics.items() if np.isscalar(value) or value is None}
    print(json.dumps({"graph_id": graph.graph_id, "group_id": graph.group_id,
                      **metadata, "metrics": summary}, ensure_ascii=False, indent=2))
