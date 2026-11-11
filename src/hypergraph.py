"""混合超图及唯一结构指标实现，定义见研究总方案附录 A。"""

from collections import Counter
from dataclasses import dataclass
from itertools import combinations

import networkx as nx
import numpy as np


def _edge_table(edges, width, N):
    table = np.asarray(edges)
    if table.shape == (0,):
        table = np.empty((0, width), dtype=np.int32)
    if table.ndim != 2 or table.shape[1] != width:
        raise ValueError(f"边表形状必须为 [M, {width}]")
    if table.size and not np.issubdtype(table.dtype, np.integer):
        raise ValueError("节点 ID 必须为整数")
    if np.any(table < 0) or np.any(table >= N):
        raise ValueError("节点 ID 超出 [0, N) 范围")
    table = np.sort(table, axis=1).astype(np.int32)
    if np.any(np.diff(table, axis=1) == 0):
        raise ValueError("一条边内不能有重复节点")
    table = table[np.lexsort(table[:, ::-1].T)]
    if len(np.unique(table, axis=0)) != len(table):
        raise ValueError("边表不能包含重复边")
    table.setflags(write=False)
    return table


@dataclass(frozen=True, eq=False)
class Graph:
    """静态混合超图；初始化规范排序后，行号即固定的原始超边 ID。

    删除由外部 bool [M0] 的 alive_mask 表达，不修改或重新编号边表。
    """

    graph_id: str
    group_id: str
    N: int
    edge_index_2: np.ndarray
    hyperedge_nodes_3: np.ndarray

    def __post_init__(self):
        if not isinstance(self.N, (int, np.integer)) or isinstance(self.N, bool):
            raise ValueError("N 必须为正整数")
        if not 0 < self.N <= np.iinfo(np.int32).max:
            raise ValueError("N 必须在 int32 正整数范围内")
        object.__setattr__(self, "edge_index_2", _edge_table(self.edge_index_2, 2, self.N))
        object.__setattr__(
            self, "hyperedge_nodes_3", _edge_table(self.hyperedge_nodes_3, 3, self.N)
        )

    @property
    def M2(self):
        return len(self.edge_index_2)

    @property
    def M0(self):
        return len(self.hyperedge_nodes_3)

    def alive_edge_ids(self, alive_mask=None):
        if alive_mask is None:
            return np.arange(self.M0)
        mask = np.asarray(alive_mask)
        if mask.shape != (self.M0,) or mask.dtype != np.bool_:
            raise ValueError("alive_mask 必须为 bool [M0]")
        return np.flatnonzero(mask)

    def adjacency_matrix(self):
        """按需生成固定二元骨架 A，不补充三元超边的二元面。"""
        adjacency = np.zeros((self.N, self.N), dtype=np.int32)
        i, j = self.edge_index_2.T
        adjacency[i, j] = adjacency[j, i] = 1
        return adjacency

    def incidence_matrix(self, alive_mask=None):
        """返回 H [N,M0]，被删超边列为零，原始列号不变。"""
        ids = self.alive_edge_ids(alive_mask)
        incidence = np.zeros((self.N, self.M0), dtype=np.int32)
        incidence[self.hyperedge_nodes_3[ids].ravel(), np.repeat(ids, 3)] = 1
        return incidence

    def pair_incidence_matrix(self, alive_mask=None):
        """返回 (node_pair_index [Pt,2], B [Pt,M0])，只保留当前使用的 Pair。"""
        ids = self.alive_edge_ids(alive_mask)
        pairs, rows, columns, _ = _pair_index(self, ids)
        incidence = np.zeros((len(pairs), self.M0), dtype=np.int32)
        incidence[rows, columns] = 1
        return pairs, incidence


def _pair_index(graph, ids):
    edges = graph.hyperedge_nodes_3[ids]
    members = np.concatenate([edges[:, [0, 1]], edges[:, [0, 2]], edges[:, [1, 2]]])
    pairs, rows, counts = np.unique(
        members, axis=0, return_inverse=True, return_counts=True
    )
    return pairs, rows, np.tile(ids, 3), counts


def overlap_graphs(graph, alive_mask=None):
    """返回 G1/G2；顶点为存活的原始超边 ID，连边条件为共享节点数 >=1/>=2。"""
    ids = graph.alive_edge_ids(alive_mask)
    incident = {}
    for edge_id in ids:
        for node in graph.hyperedge_nodes_3[edge_id]:
            incident.setdefault(int(node), []).append(int(edge_id))
    overlap = Counter()
    for edge_ids in incident.values():
        overlap.update(combinations(edge_ids, 2))
    one, two = nx.Graph(), nx.Graph()
    one.add_nodes_from(map(int, ids))
    two.add_nodes_from(map(int, ids))
    one.add_edges_from(overlap)
    two.add_edges_from(pair for pair, count in overlap.items() if count >= 2)
    return one, two


def compute_metrics(graph, alive_mask=None, community_labels=None):
    """计算当前结构指标，不构造稠密 A/H/B。

    d2/d3 按节点 ID 对齐，o1/o2 按原始超边 ID 对齐（被删项为零）；
    q_ij 与 node_pair_index 的行对齐。空三元层的 xi_A 记为 None (NA)。
    rho_cross 在缺少社区标签或三元层为空时也记为 None。
    """
    ids = graph.alive_edge_ids(alive_mask)
    edges = graph.hyperedge_nodes_3[ids]
    M = len(ids)
    d2 = np.bincount(graph.edge_index_2.ravel(), minlength=graph.N)
    d3 = np.bincount(edges.ravel(), minlength=graph.N)
    pairs, _, _, q_ij = _pair_index(graph, ids)
    one, two = overlap_graphs(graph, alive_mask)
    o1 = np.zeros(graph.M0, dtype=np.int64)
    o2 = np.zeros(graph.M0, dtype=np.int64)
    for edge_id in ids:
        o2[edge_id] = two.degree(edge_id)
        # 统计量 o1 恰好共享一个节点，不能直接使用 G1 的度。
        o1[edge_id] = one.degree(edge_id) - o2[edge_id]

    # G1 最大分量按覆盖的原始节点数选取，而非按超边数选取。
    largest_node_cover = max(
        (len(np.unique(graph.hyperedge_nodes_3[list(c)])) for c in nx.connected_components(one)),
        default=0,
    )
    largest_edge_count = max((len(c) for c in nx.connected_components(two)), default=0)
    binary_edges = set(map(tuple, graph.edge_index_2))
    binary_pair_occurrences = sum(
        int(count) for pair, count in zip(pairs, q_ij) if tuple(pair) in binary_edges
    )
    rho_cross = None
    if community_labels is not None:
        labels = np.asarray(community_labels)
        if labels.shape != (graph.N,) or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("community_labels 必须为长度 N 的整数数组")
        if M:
            rho_cross = float(np.mean(np.any(labels[edges] != labels[edges[:, [0]]], axis=1)))
    C2 = two.number_of_edges()
    return {
        "N": graph.N,
        "M2": graph.M2,
        "M0": graph.M0,
        "M3": M,
        "d2": d2,
        "d3": d3,
        "node_pair_index": pairs,
        "q_ij": q_ij,
        "o1": o1,
        "o2": o2,
        "C1_exact": one.number_of_edges() - C2,
        "C2": C2,
        "f2": float(np.count_nonzero(o2) / M) if M else 0.0,
        "xi_A": binary_pair_occurrences / (3 * M) if M else None,
        "rho_cross": rho_cross,
        "L1": largest_node_cover / graph.N,
        # 分母始终是初始 M0；删除后不能改为当前存活数 M。
        "L2": largest_edge_count / graph.M0 if graph.M0 else 0.0,
        "B1": sum(1 for _ in nx.articulation_points(one)),
        "B2": sum(1 for _ in nx.articulation_points(two)),
    }
