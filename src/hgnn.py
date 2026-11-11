"""Node–Pair–Hyperedge 编码与完整结构距离偏置 Attention。"""

from functools import lru_cache
from copy import deepcopy

import networkx as nx
import numpy as np
import torch
from torch import nn

from .hypergraph import compute_metrics, overlap_graphs


def mlp(input_dim, output_dim):
    return nn.Sequential(nn.Linear(input_dim, output_dim), nn.ReLU(),
                         nn.Linear(output_dim, output_dim))


def _sum(values, index, size):
    return values.new_zeros((size, values.shape[-1])).index_add(0, index, values)


def _pool(values, members, aggregation):
    selected = values[members]
    return selected.sum(1) if aggregation == "sum_mlp" else selected.mean(1)


ABLATIONS = ("full", "no_pair", "no_distance_bias", "no_global_attention", "mean_aggregation")


def ablation_config(config, name):
    if name not in ABLATIONS:
        raise ValueError("未知消融名称")
    result = deepcopy(config)
    # 每次从同一完整架构出发，只改变一个因素。
    result["hgnn"].update(use_pair=True, use_distance_bias=True,
                          use_global_attention=True, aggregation="sum_mlp")
    changes = {"full": {}, "no_pair": {"use_pair": False},
               "no_distance_bias": {"use_distance_bias": False},
               "no_global_attention": {"use_global_attention": False},
               "mean_aggregation": {"aggregation": "mean_mlp"}}
    result["hgnn"].update(changes[name])
    result["ablation"] = name
    return result


class LocalLayer(nn.Module):
    def __init__(self, dim, use_pair=True, aggregation="sum_mlp"):
        super().__init__()
        self.use_pair, self.aggregation = use_pair, aggregation
        self.node_edge = mlp(dim, dim)
        self.edge_update = mlp((3 if use_pair else 2) * dim, dim)
        self.edge_node, self.binary_node = mlp(dim, dim), mlp(dim, dim)
        self.node_update = mlp(3 * dim, dim)
        self.edge_norm, self.node_norm = nn.LayerNorm(dim), nn.LayerNorm(dim)
        if use_pair:
            self.node_pair = mlp(4 * dim, dim)
            self.pair_edge = mlp(dim, dim)
            self.edge_pair = mlp(dim, dim)
            self.pair_update = mlp(2 * dim, dim)
            self.pair_pre_norm, self.pair_norm = nn.LayerNorm(dim), nn.LayerNorm(dim)

    def forward(self, h, p, z, members, pair_members, pairs, binary, d2, d3, q):
        node_message = _pool(self.node_edge(h), members, self.aggregation)
        messages = [z, node_message]
        if self.use_pair:
            i, j = pairs.T
            p_bar = self.pair_pre_norm(p + self.node_pair(torch.cat(
                (p, h[i] + h[j], (h[i] - h[j]).abs(), h[i] * h[j]), dim=-1)))
            messages.append(_pool(self.pair_edge(p_bar), pair_members, self.aggregation))
        z = self.edge_norm(z + self.edge_update(torch.cat(messages, dim=-1)))
        # 同一轮的节点更新仍使用旧 h；超边返回消息使用更新后的 z。
        edge_message = _sum(self.edge_node(z).repeat_interleave(3, 0), members.flatten(), len(h))
        source = torch.cat((binary[:, 0], binary[:, 1]))
        target = torch.cat((binary[:, 1], binary[:, 0]))
        binary_message = _sum(self.binary_node(h)[source], target, len(h))
        h = self.node_norm(h + self.node_update(torch.cat(
            (h, edge_message / d3.clamp_min(1)[:, None],
             binary_message / d2.clamp_min(1)[:, None]), dim=-1)))
        if self.use_pair:
            pair_message = _sum(self.edge_pair(z).repeat_interleave(3, 0), pair_members.flatten(), len(p))
            p = self.pair_norm(p_bar + self.pair_update(torch.cat(
                (p_bar, pair_message / q.clamp_min(1)[:, None]), dim=-1)))
        return h, p, z


class LocalHGNN(nn.Module):
    def __init__(self, hidden_dim=64, local_layers=2, use_pair=True, aggregation="sum_mlp"):
        super().__init__()
        if hidden_dim < 1 or local_layers < 1:
            raise ValueError("hidden_dim 和 local_layers 必须为正整数")
        self.hidden_dim = hidden_dim
        if aggregation not in ("sum_mlp", "mean_mlp"):
            raise ValueError("aggregation 必须为 sum_mlp 或 mean_mlp")
        self.use_pair, self.aggregation = use_pair, aggregation
        self.node_init = mlp(2, hidden_dim)
        self.node_init_edge = mlp(hidden_dim, hidden_dim)
        if use_pair:
            self.pair_init = mlp(2 * hidden_dim + 2, hidden_dim)
            self.pair_init_edge = mlp(hidden_dim, hidden_dim)
        self.edge_init = mlp((2 if use_pair else 1) * hidden_dim + 2, hidden_dim)
        self.layers = nn.ModuleList(LocalLayer(hidden_dim, use_pair, aggregation) for _ in range(local_layers))

    def forward(self, graph, alive_mask=None):
        ids = graph.alive_edge_ids(alive_mask)
        if not len(ids):
            raise ValueError("终止空状态不进入 HGNN")
        metrics = compute_metrics(graph, alive_mask)
        device, dtype = next(self.parameters()).device, next(self.parameters()).dtype

        def index(value):
            return torch.tensor(np.asarray(value), dtype=torch.long, device=device)

        def feature(value):
            return torch.tensor(np.asarray(value), dtype=dtype, device=device)

        members = index(graph.hyperedge_nodes_3[ids])
        pairs = metrics["node_pair_index"]
        lookup = {tuple(pair): i for i, pair in enumerate(pairs)}
        pair_members = index([[lookup[(int(a), int(b))] for a, b in
                               ((edge[0], edge[1]), (edge[0], edge[2]), (edge[1], edge[2]))]
                              for edge in graph.hyperedge_nodes_3[ids]])
        binary = index(graph.edge_index_2)
        d2, d3, q = (feature(metrics[key]) for key in ("d2", "d3", "q_ij"))
        h = self.node_init(torch.stack((d2.log1p(), d3.log1p()), dim=-1))
        binary_set = set(map(tuple, graph.edge_index_2))
        pair_binary = feature([tuple(pair) in binary_set for pair in pairs])
        pairs = index(pairs)
        i, j = pairs.T
        p = (self.pair_init(torch.cat((h[i] + h[j], (h[i] - h[j]).abs(),
                                       pair_binary[:, None], q.log1p()[:, None]), dim=-1))
             if self.use_pair else h.new_zeros((len(pairs), self.hidden_dim)))
        overlap = feature(np.stack((metrics["o1"][ids], metrics["o2"][ids]), axis=-1))
        messages = [_pool(self.node_init_edge(h), members, self.aggregation)]
        if self.use_pair:
            messages.append(_pool(self.pair_init_edge(p), pair_members, self.aggregation))
        z = self.edge_init(torch.cat((*messages, overlap), dim=-1))
        for layer in self.layers:
            h, p, z = layer(h, p, z, members, pair_members, pairs, binary, d2, d3, q)
        return {"node_embeddings": h, "pair_embeddings": p, "edge_embeddings": z,
                "edge_ids": index(ids), "node_pair_index": pairs,
                "members": members, "pair_members": pair_members}


@lru_cache(maxsize=128)
def binary_edge_distances(graph):
    """只缓存固定 A 上的成员集合最小距离；缓存键为不可变 Graph 对象。"""
    network = nx.Graph()
    network.add_nodes_from(range(graph.N))
    network.add_edges_from(graph.edge_index_2)
    node_distances = np.full((graph.N, graph.N), np.inf)
    for node, lengths in nx.all_pairs_shortest_path_length(network):
        for other, distance in lengths.items():
            node_distances[node, other] = distance
    edges = graph.hyperedge_nodes_3
    distances = node_distances[edges[:, None, :, None], edges[None, :, None, :]].min(axis=(2, 3))
    distances.setflags(write=False)
    return distances


def distance_buckets(distances, distance_max=6):
    if not isinstance(distance_max, int) or distance_max < 0:
        raise ValueError("distance_max 必须为非负整数")
    return np.where(np.isfinite(distances), np.minimum(distances, distance_max + 1),
                    distance_max + 2).astype(np.int64)


def structure_distances(graph, alive_mask=None, distance_max=6):
    ids = graph.alive_edge_ids(alive_mask)
    positions = {int(edge): i for i, edge in enumerate(ids)}
    distances = []
    # D1/D2 在当前诱导辅助图上重算，不能截取初始最短路径矩阵。
    for network in overlap_graphs(graph, alive_mask):
        matrix = np.full((len(ids), len(ids)), np.inf)
        for edge, lengths in nx.all_pairs_shortest_path_length(network):
            for other, length in lengths.items():
                matrix[positions[edge], positions[other]] = length
        distances.append(distance_buckets(matrix, distance_max))
    distances.append(distance_buckets(binary_edge_distances(graph)[np.ix_(ids, ids)], distance_max))
    return tuple(distances)


class StructuralAttention(nn.Module):
    def __init__(self, dim, heads, distance_max, use_distance_bias=True):
        super().__init__()
        if heads < 1 or dim % heads:
            raise ValueError("hidden_dim 必须能被 heads 整除")
        self.heads, self.head_dim = heads, dim // heads
        self.qkv, self.output = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.bias = nn.ModuleList(nn.Embedding(distance_max + 3, heads) for _ in range(3)) if use_distance_bias else nn.ModuleList()

    def forward(self, x, buckets):
        q, k, v = self.qkv(x).reshape(len(x), 3, self.heads, self.head_dim).unbind(1)
        q, k, v = (value.transpose(0, 1) for value in (q, k, v))
        score = q @ k.transpose(-1, -2) / self.head_dim**0.5
        for table, distance in zip(self.bias, buckets):
            score = score + table(distance).permute(2, 0, 1)
        # 输入已压缩为单图存活超边；UNREACHABLE 仍参与全局 Attention。
        weights = score.softmax(dim=-1)
        output = (weights @ v).transpose(0, 1).reshape(len(x), -1)
        return self.output(output), weights


class GlobalLayer(nn.Module):
    def __init__(self, dim, heads, distance_max, dropout=0.0, use_distance_bias=True):
        super().__init__()
        self.attention = StructuralAttention(dim, heads, distance_max, use_distance_bias)
        self.attention_norm, self.ffn_norm, self.output_norm = (nn.LayerNorm(dim) for _ in range(3))
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.ReLU(), nn.Linear(4 * dim, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, buckets):
        attended, _ = self.attention(self.attention_norm(x), buckets)
        y = x + self.dropout(attended)
        return self.output_norm(y + self.dropout(self.ffn(self.ffn_norm(y))))


class HGNN(nn.Module):
    def __init__(self, hidden_dim=64, local_layers=2, heads=4, global_layers=1, distance_max=6,
                 use_pair=True, use_distance_bias=True, use_global_attention=True, aggregation="sum_mlp"):
        super().__init__()
        if global_layers < 1 or distance_max < 0:
            raise ValueError("global_layers 必须为正数，distance_max 必须非负")
        self.hidden_dim, self.distance_max = hidden_dim, distance_max
        self.use_distance_bias = use_distance_bias
        self.use_pair, self.aggregation = use_pair, aggregation
        self.local = LocalHGNN(hidden_dim, local_layers, use_pair, aggregation)
        self.node_pool = mlp(hidden_dim, hidden_dim)
        if use_pair:
            self.pair_pool = mlp(hidden_dim, hidden_dim)
        self.fusion = nn.Linear((2 if use_pair else 1) * hidden_dim, hidden_dim)
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.global_layers = nn.ModuleList(GlobalLayer(hidden_dim, heads, distance_max,
                                                        use_distance_bias=use_distance_bias)
                                            for _ in range(global_layers if use_global_attention else 0))
        self.readout = mlp(2 * hidden_dim + 2, hidden_dim)

    def forward(self, graph, alive_mask=None):
        local = self.local(graph, alive_mask)
        ids, z = local["edge_ids"], local["edge_embeddings"]
        pools = [_pool(self.node_pool(local["node_embeddings"]), local["members"], self.aggregation)]
        if self.use_pair:
            pools.append(_pool(self.pair_pool(local["pair_embeddings"]), local["pair_members"], self.aggregation))
        x = self.fusion_norm(z + self.fusion(torch.cat(pools, dim=-1)))
        if self.global_layers:
            buckets = (tuple(torch.tensor(d, device=x.device, dtype=torch.long)
                             for d in structure_distances(graph, alive_mask, self.distance_max))
                       if self.use_distance_bias else ())
            for layer in self.global_layers:
                x = layer(x, buckets)
        sizes = x.new_tensor(np.log1p([graph.N, len(ids)]))
        global_embedding = self.readout(torch.cat((x.mean(0), x.amax(0), sizes)))
        assert x.shape == (len(ids), self.hidden_dim) and global_embedding.shape == (self.hidden_dim,)
        embeddings = x.new_zeros((graph.M0, self.hidden_dim)).index_copy(0, ids, x)
        valid_mask = torch.zeros(graph.M0, device=x.device, dtype=torch.bool)
        valid_mask[ids] = True
        return embeddings, global_embedding, valid_mask

    def forward_batch(self, items):
        """逐图编码可变规模 Batch，图之间绝不共享 Attention。"""
        return [self(graph, mask) for graph, mask in items]

    @classmethod
    def from_config(cls, config):
        names = ("hidden_dim", "local_layers", "heads", "global_layers", "distance_max")
        parameters = {name: config["hgnn"][name] for name in names}
        for name, default in (("use_pair", True), ("use_distance_bias", True),
                              ("use_global_attention", True), ("aggregation", "sum_mlp")):
            parameters[name] = config["hgnn"].get(name, default)
        return cls(**parameters)
