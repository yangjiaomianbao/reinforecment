"""原始混合超图的 NPZ/manifest 读写与祖先家族隔离划分。"""

import argparse
import csv
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import networkx as nx
import numpy as np

from .hypergraph import Graph, compute_metrics


SPLITS = ("train", "validation", "test_id", "ood_size", "ood_structure", "calibration")


def _mixed_network(graph):
    network = nx.Graph()
    network.add_nodes_from(range(graph.N), kind="node")
    network.add_edges_from(graph.edge_index_2)
    network.add_nodes_from(range(graph.N, graph.N + graph.M0), kind="hyperedge")
    for edge_id, nodes in enumerate(graph.hyperedge_nodes_3):
        network.add_edges_from((int(node), graph.N + edge_id) for node in nodes)
    return network


def _families(samples):
    """合并祖先、相同 group_id 和同构骨架；精确同构检查拒绝完整拓扑重复。"""
    parent = {g.group_id: g.group_id for g, _ in samples}
    by_id = {}
    for graph, metadata in samples:
        if graph.graph_id in by_id:
            raise ValueError(f"DUPLICATE_GRAPH_ID: {graph.graph_id}")
        by_id[graph.graph_id] = graph

    def root(group):
        while parent[group] != group:
            parent[group] = parent[parent[group]]
            group = parent[group]
        return group

    def merge(a, b):
        a, b = root(a), root(b)
        parent[max(a, b)] = min(a, b)

    ancestors, skeletons, topologies = {}, {}, {}
    for graph, metadata in samples:
        ancestor = metadata.get("ancestor_id", graph.graph_id)
        if ancestor in by_id:
            merge(graph.group_id, by_id[ancestor].group_id)
        if ancestor in ancestors:
            merge(graph.group_id, ancestors[ancestor])
        ancestors[ancestor] = graph.group_id
        d2 = np.bincount(graph.edge_index_2.ravel(), minlength=graph.N)
        d3 = np.bincount(graph.hyperedge_nodes_3.ravel(), minlength=graph.N)
        signature = (graph.N, graph.M2, tuple(sorted(map(int, d2))))
        binary = nx.Graph()
        binary.add_nodes_from(range(graph.N))
        binary.add_edges_from(graph.edge_index_2)
        bucket = skeletons.setdefault(signature, [])
        for other, network in bucket:
            if nx.is_isomorphic(binary, network):
                merge(graph.group_id, other.group_id)
                break
        else:
            bucket.append((graph, binary))
        # 度签名仅用于筛选候选；不将签名相同直接视为同构。
        signature = (graph.N, graph.M2, graph.M0,
                     tuple(sorted(zip(map(int, d2), map(int, d3)))))
        mixed = _mixed_network(graph)
        bucket = topologies.setdefault(signature, [])
        for other, network in bucket:
            if nx.is_isomorphic(mixed, network, node_match=lambda a, b: a["kind"] == b["kind"]):
                raise ValueError(f"DUPLICATE_TOPOLOGY: {other.graph_id}, {graph.graph_id}")
        bucket.append((graph, mixed))
    return {graph.graph_id: root(graph.group_id) for graph, _ in samples}


def assign_splits(samples, weights, *, seed, fixed_splits=None):
    """返回 graph_id -> split；按家族分配而非按图随机打散。

    weights 仅用于 ID 划分；OOD/Calibration 必须通过 group_id -> split 显式指定。
    家族足够时每个 ID split 先分配一个家族，余下按权重最大余数法分配。
    因家族大小不同，这不是正式图数量的精确配额。
    """
    samples = list(samples)
    families = _families(samples)
    fixed_splits = {} if fixed_splits is None else fixed_splits
    groups = {g.group_id for g, _ in samples}
    if set(fixed_splits) - groups:
        raise ValueError("fixed_splits 包含不存在的 group_id")
    if any(split not in SPLITS for split in fixed_splits.values()):
        raise ValueError("未知 split")
    names = list(weights)
    values = np.asarray(list(weights.values()), dtype=float)
    if (not names or any(name not in SPLITS[:3] for name in names)
            or not np.all(np.isfinite(values)) or np.any(values <= 0)
            or not np.isfinite(values.sum())):
        raise ValueError("weights 必须为 train/validation/test_id 的有限正权重")
    assigned = {}
    for graph, _ in samples:
        if graph.group_id in fixed_splits:
            family, split = families[graph.graph_id], fixed_splits[graph.group_id]
            if family in assigned and assigned[family] != split:
                raise ValueError("SPLIT_LEAKAGE: 同一祖先或同构骨架被指定到不同 split")
            assigned[family] = split
    remaining = sorted(set(families.values()) - set(assigned))
    order = np.random.default_rng(np.random.SeedSequence([seed, 5])).permutation(len(remaining))
    remaining = [remaining[i] for i in order]
    counts = np.zeros(len(names), dtype=int)
    # 已显式分配到某个 ID split 的家族也计入非空条件。
    missing = [i for i, name in enumerate(names) if name not in assigned.values()]
    if len(remaining) >= len(missing):
        counts[missing] = 1
    quotas = (len(remaining) - counts.sum()) * values / values.sum()
    counts += np.floor(quotas).astype(int)
    for i in np.argsort(-(quotas - np.floor(quotas)), kind="stable")[:len(remaining) - counts.sum()]:
        counts[i] += 1
    offset = 0
    for name, count in zip(names, counts):
        for family in remaining[offset:offset + count]:
            assigned[family] = name
        offset += count
    return {graph_id: assigned[family] for graph_id, family in families.items()}


def _check_splits(samples, splits, families):
    if set(splits) != set(families) or any(split not in SPLITS for split in splits.values()):
        raise ValueError("splits 必须为每张图指定一个有效划分")
    assigned = {}
    for graph, _ in samples:
        family, split = families[graph.graph_id], splits[graph.graph_id]
        if family in assigned and assigned[family] != split:
            raise ValueError("SPLIT_LEAKAGE: 同一祖先或同构骨架不能跨 split")
        assigned[family] = split


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"元数据不能序列化: {type(value).__name__}")


def save_dataset(directory, samples, splits):
    """保存原始边表及唯一 manifest.csv；写入前验证全部样本和划分，不覆盖已有数据。"""
    samples = list(samples)
    if not samples:
        raise ValueError("数据集不能为空")
    families = _families(samples)
    _check_splits(samples, splits, families)
    source = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        source.update(path.name.encode() + b"\0" + path.read_bytes())
    rows = []
    for index, (graph, metadata) in enumerate(samples):
        seed, generator_type = metadata["seed"], metadata["generator_type"]
        if not isinstance(seed, (int, np.integer)) or seed < 0 or not isinstance(generator_type, str):
            raise ValueError("元数据必须包含非负整数 seed 和字符串 generator_type")
        metrics = compute_metrics(graph, community_labels=metadata.get("community_labels"))
        row = {"graph_id": graph.graph_id, "group_id": graph.group_id,
               "family_id": families[graph.graph_id], "ancestor_id": metadata.get("ancestor_id", graph.graph_id),
               "split": splits[graph.graph_id], "file": f"{index:06d}.npz",
               "generator_type": generator_type, "seed": int(seed), "code_hash": source.hexdigest(),
               "metadata": json.dumps(metadata, default=_json_default, ensure_ascii=False, allow_nan=False)}
        row.update({key: value for key, value in metrics.items() if np.isscalar(value) or value is None})
        row.update({key: json.dumps(metrics[key].tolist()) for key in ("d2", "d3")})
        rows.append(row)
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("目标目录非空；请选择新的输出目录")
    directory.mkdir(parents=True, exist_ok=True)
    for (graph, _), row in zip(samples, rows):
        with (directory / row["file"]).open("xb") as file:
            np.savez_compressed(file, N=np.int32(graph.N), edge_index_2=graph.edge_index_2,
                                hyperedge_nodes_3=graph.hyperedge_nodes_3,
                                graph_id=graph.graph_id, group_id=graph.group_id,
                                split=row["split"], generator_type=row["generator_type"], seed=row["seed"])
    with (directory / "manifest.csv").open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def load_dataset(directory, split=None):
    """返回 [(Graph, metadata)]；先核对全表家族隔离，再按 split 筛选。"""
    if split is not None and split not in SPLITS:
        raise ValueError("未知 split")
    directory = Path(directory)
    with (directory / "manifest.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("manifest 不能为空")
    samples, splits = [], {}
    for row in rows:
        path = (directory / row["file"]).resolve()
        if path.parent != directory.resolve():
            raise ValueError("NPZ 文件必须位于数据集目录内")
        with np.load(path, allow_pickle=False) as data:
            graph = Graph(str(data["graph_id"].item()), str(data["group_id"].item()),
                          int(data["N"].item()), data["edge_index_2"], data["hyperedge_nodes_3"])
            for field in ("graph_id", "group_id", "split", "generator_type", "seed"):
                if str(data[field].item()) != row[field]:
                    raise ValueError(f"NPZ 与 manifest 的 {field} 不一致")
            # 不允许读取时通过重新排序悄悄改变原始超边 ID。
            if not (np.array_equal(graph.edge_index_2, data["edge_index_2"])
                    and np.array_equal(graph.hyperedge_nodes_3, data["hyperedge_nodes_3"])):
                raise ValueError("NPZ 边表未按规范排序")
        metadata = json.loads(row["metadata"])
        if metadata["seed"] != int(row["seed"]) or metadata["generator_type"] != row["generator_type"]:
            raise ValueError("manifest 元数据不一致")
        metadata.update(split=row["split"], family_id=row["family_id"], code_hash=row["code_hash"])
        samples.append((graph, metadata))
        splits[graph.graph_id] = row["split"]
    families = _families(samples)
    _check_splits(samples, splits, families)
    return [(graph, metadata) for graph, metadata in samples if split is None or metadata["split"] == split]


if __name__ == "__main__":
    from .generators import generate_graph
    from .rewiring import rewire
    from .utils import load_config

    parser = argparse.ArgumentParser(description="少量祖先组的数据读写与隔离划分检查")
    parser.add_argument("--config", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--groups", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--rewired", action="store_true", help="每组额外尝试一个固定超度重连变体")
    args = parser.parse_args()
    config = load_config(args.config)
    groups = config["dataset"]["smoke_groups"] if args.groups is None else args.groups
    if groups < 1:
        parser.error("--groups 必须为正整数")
    seed = config["seed"] if args.seed is None else args.seed
    samples, outcomes = [], []
    for child in np.random.SeedSequence([seed, 6]).spawn(groups):
        graph_seed, rewire_seed = map(int, child.generate_state(2))
        ancestor, metadata = generate_graph(config, seed=graph_seed)
        if args.rewired:
            variant, diagnostic = rewire(ancestor, seed=rewire_seed, attempts=3 * ancestor.M0)
            diagnostic.pop("C2_history")
            metadata["rewiring_attempt"] = diagnostic
            distinct = not np.array_equal(ancestor.hyperedge_nodes_3, variant.hyperedge_nodes_3)
            outcomes.append({"ancestor_id": ancestor.graph_id,
                             "status": "DISTINCT_VARIANT" if distinct else "NO_DISTINCT_VARIANT_OBSERVED"})
            if distinct:
                variant_metadata = deepcopy(metadata)
                variant_metadata.update(seed=rewire_seed, ancestor_id=ancestor.graph_id,
                                        generator_type=metadata["generator_type"] + "/DegreePreservingRewire")
                samples.append((variant, variant_metadata))
        samples.append((ancestor, metadata))
    weights = {name: config["dataset"]["candidate_counts"][name] for name in SPLITS[:3]}
    splits = assign_splits(samples, weights, seed=seed)
    rows = save_dataset(args.output, samples, splits)
    restored = load_dataset(args.output)
    print(json.dumps({"graphs": len(restored), "groups": groups,
                      "split_counts": {name: sum(row["split"] == name for row in rows) for name in SPLITS[:3]},
                      "rewiring_outcomes": outcomes, "manifest": str(Path(args.output) / "manifest.csv"),
                      "science_status": "NOT_EVALUATED"}, ensure_ascii=False, indent=2))
