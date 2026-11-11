"""固定逐节点三元超度的成员交换、无偏探索和目标 C2 重连。"""

import argparse
import hashlib
import json

import numpy as np

from .hypergraph import Graph, compute_metrics


def swap_members(graph, edge_a, edge_b, member_a, member_b):
    """交换两边各自独有的成员；重复边或拓扑不变的提案返回 None。

    返回新的原始图，其三元边重新规范排序；它不是删边过程中的 State。
    祖先 graph 本身不变，所有重连变体保留相同 group_id。
    """
    if not (0 <= edge_a < graph.M0 and 0 <= edge_b < graph.M0) or edge_a == edge_b:
        raise ValueError("必须选择两条不同且存在的三元超边")
    a, b = (set(map(int, graph.hyperedge_nodes_3[e])) for e in (edge_a, edge_b))
    if member_a not in a - b or member_b not in b - a:
        raise ValueError("互换成员必须分别属于两条边的独有成员")
    old_a, old_b = tuple(sorted(a)), tuple(sorted(b))
    new_a = tuple(sorted((a - {member_a}) | {member_b}))
    new_b = tuple(sorted((b - {member_b}) | {member_a}))
    if {old_a, old_b} == {new_a, new_b}:
        return None
    edges = set(map(tuple, graph.hyperedge_nodes_3)) - {old_a, old_b}
    if new_a == new_b or new_a in edges or new_b in edges:
        return None
    triples = np.asarray(sorted(edges | {new_a, new_b}), dtype=np.int32)
    digest = hashlib.sha256(np.asarray([graph.N], dtype="<i8").tobytes()
                            + graph.edge_index_2.astype("<i4").tobytes()
                            + triples.astype("<i4").tobytes())
    return Graph("graph_" + digest.hexdigest()[:16], graph.group_id,
                 graph.N, graph.edge_index_2, triples)


def rewire(graph, *, seed, attempts, target_C2=None, temperature=1.0, tolerance=1):
    """返回 (结果图, 诊断)。无目标时返回链末态，有目标时返回误差最小的已访状态。

    对称提案：均匀选择边对，再均匀选择各自独有成员；无效提案保持原地。
    有目标时能量为 |C2-target_C2|，按 min(1,exp(-ΔE/T)) 接受。
    尽量精确命中整数目标；预算耗尽后才按 tolerance 判断是否达标。
    """
    if not isinstance(attempts, int) or attempts < 0:
        raise ValueError("attempts 必须为非负整数")
    if target_C2 is not None and (not isinstance(target_C2, (int, np.integer)) or target_C2 < 0):
        raise ValueError("target_C2 必须为非负整数或 None")
    if not np.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature 必须为有限非负数")
    if not isinstance(tolerance, int) or tolerance < 0:
        raise ValueError("tolerance 必须为非负整数")
    rng = np.random.default_rng(seed)
    current, best = graph, graph
    current_C2 = best_C2 = compute_metrics(graph)["C2"]
    history = [current_C2]
    valid = accepted = 0
    first_swap = None
    for _ in range(attempts):
        if graph.M0 < 2 or (target_C2 is not None and current_C2 == target_C2):
            break
        i, j = rng.choice(graph.M0, 2, replace=False)
        a, b = (set(map(int, current.hyperedge_nodes_3[e])) for e in (i, j))
        candidate = None
        if a - b and b - a:
            candidate = swap_members(current, i, j,
                                     int(rng.choice(sorted(a - b))),
                                     int(rng.choice(sorted(b - a))))
        if candidate is not None:
            valid += 1
            candidate_C2 = compute_metrics(candidate)["C2"]
            delta = (abs(candidate_C2 - target_C2) - abs(current_C2 - target_C2)
                     if target_C2 is not None else 0)
            accept = delta <= 0 or (temperature > 0 and rng.random() < np.exp(-delta / temperature))
            if accept:
                if first_swap is None:
                    first_swap = {"C2_before": current_C2, "C2_after": candidate_C2}
                current, current_C2 = candidate, candidate_C2
                accepted += 1
                if target_C2 is None or abs(current_C2 - target_C2) < abs(best_C2 - target_C2):
                    best, best_C2 = current, current_C2
        # 拒绝也记一次停留，不能用仅含成功交换的跳跃链估计经验分位点。
        history.append(current_C2)
    error = abs(best_C2 - target_C2) if target_C2 is not None else None
    status = ("TARGET_REACHED" if error <= tolerance else "TARGET_NOT_REACHED") if error is not None else "SAMPLING_COMPLETE"
    if target_C2 is None and accepted == 0:
        status = "NO_VALID_SWAP_OBSERVED"
    return best, {
        "ancestor_id": graph.graph_id, "seed": int(seed), "status": status,
        "target_C2": int(target_C2) if target_C2 is not None else None,
        "actual_C2": best_C2, "target_error": error,
        "temperature": temperature, "tolerance": tolerance,
        "attempts": len(history) - 1, "valid_proposals": valid,
        "accepted": accepted, "first_swap": first_swap, "C2_history": history,
    }


def explore_overlap(graph, *, seed, chains=4, attempts_per_edge=200):
    """四条或更多独立无偏链；返回观测整数集合及经验 10%/50%/90% 目标。"""
    if not isinstance(chains, int) or chains < 4:
        raise ValueError("可行性探索至少需要四条独立链")
    if not isinstance(attempts_per_edge, int) or attempts_per_edge < 1:
        raise ValueError("attempts_per_edge 必须为正整数")
    reports, samples = [], []
    for child in np.random.SeedSequence([seed, 3]).spawn(chains):
        _, report = rewire(graph, seed=int(child.generate_state(1)[0]),
                           attempts=attempts_per_edge * graph.M0)
        reports.append(report)
        samples.extend(report["C2_history"])
    observed = sorted(set(samples))
    quantiles = np.quantile(samples, [.1, .5, .9]).tolist()
    targets = [min(observed, key=lambda value: (abs(value - q), value)) for q in quantiles]
    return {
        "status": "OVERLAP_RANGE_OBSERVED" if len(set(targets)) == 3 else "INSUFFICIENT_OVERLAP_RANGE",
        "observed_C2": observed, "empirical_quantiles": quantiles,
        "target_C2": dict(zip(("weak", "medium", "strong"), targets)),
        "chains": reports,
    }


def compare_organization(first, second, xi_A_tolerance=0.05):
    """严格相同 C2 下比较 L2/B2，同时单独检查二元–三元关联协变量。"""
    if not np.isfinite(xi_A_tolerance) or xi_A_tolerance < 0:
        raise ValueError("xi_A_tolerance 必须为有限非负数")
    a, b = compute_metrics(first), compute_metrics(second)
    fixed = (first.N == second.N and first.M0 == second.M0
             and first.group_id == second.group_id
             and np.array_equal(first.edge_index_2, second.edge_index_2)
             and np.array_equal(a["d3"], b["d3"]))
    if not fixed:
        raise ValueError("INVALID: 配对必须保持 N、M2、M3、A、逐节点超度和 group_id 相同")
    delta_L2, delta_B2 = abs(a["L2"] - b["L2"]), abs(a["B2"] - b["B2"])
    matched_C2 = a["C2"] == b["C2"]
    separated = bool(first.M0 and (delta_L2 + 1e-12 >= 2 / first.M0 or delta_B2 >= 1))
    delta_xi = abs(a["xi_A"] - b["xi_A"]) if a["xi_A"] is not None else None
    xi_matched = delta_xi <= xi_A_tolerance + 1e-12 if delta_xi is not None else None
    if not matched_C2:
        status = "C2_NOT_MATCHED"
    elif not separated:
        status = "ORGANIZATION_NOT_SEPARATED"
    elif xi_matched is not True:
        status = "COVARIATE_NOT_MATCHED"
    else:
        status = "ORGANIZATION_SEPARATED"
    return {"status": status, "C2": [a["C2"], b["C2"]],
            "delta_L2": delta_L2, "delta_B2": delta_B2,
            "delta_xi_A": delta_xi, "xi_A_matched": xi_matched,
            "organization_separated": matched_C2 and separated}


def generate_matched_variants(graph, config=None, seed=None):
    """只处理一个祖先组；返回 (三档候选图字典, 真实成功/失败诊断)。"""
    from .utils import load_config

    config = load_config() if config is None else config
    seed = config["seed"] if seed is None else seed
    p = config["rewiring"]
    target_budget = p["target_attempts_per_edge"]
    if not isinstance(target_budget, int) or target_budget < 1:
        raise ValueError("target_attempts_per_edge 必须为正整数")
    seeds = [int(child.generate_state(1)[0]) for child in np.random.SeedSequence([seed, 4]).spawn(4)]
    exploration = explore_overlap(graph, seed=seeds[0], chains=p["exploration_chains"],
                                  attempts_per_edge=p["exploration_attempts_per_edge"])
    report = {"ancestor_id": graph.graph_id, "seed": int(seed),
              "exploration": exploration, "variants": {},
              "status": exploration["status"], "science_status": "NOT_EVALUATED"}
    if exploration["status"] == "INSUFFICIENT_OVERLAP_RANGE":
        return {}, report
    variants = {}
    ancestor_metrics = compute_metrics(graph)
    for (name, target), child_seed in zip(exploration["target_C2"].items(), seeds[1:]):
        result, diagnostic = rewire(graph, seed=child_seed, attempts=target_budget * graph.M0,
                                    target_C2=target, temperature=p["temperature"],
                                    tolerance=p["target_tolerance"])
        metrics = compute_metrics(result)
        delta_xi = (abs(metrics["xi_A"] - ancestor_metrics["xi_A"])
                    if metrics["xi_A"] is not None else None)
        diagnostic.update({"graph_id": result.graph_id, "L2": metrics["L2"], "B2": metrics["B2"],
                           "delta_xi_A_from_ancestor": delta_xi,
                           "xi_A_matched_to_ancestor": delta_xi <= p["xi_A_tolerance"] + 1e-12
                           if delta_xi is not None else None})
        variants[name], report["variants"][name] = result, diagnostic
    actual = [d["actual_C2"] for d in report["variants"].values()]
    if any(d["status"] != "TARGET_REACHED" for d in report["variants"].values()):
        report["status"] = "TARGET_NOT_REACHED"
    elif not actual[0] < actual[1] < actual[2]:
        report["status"] = "TARGETS_NOT_SEPARATED"
    else:
        report["status"] = "OVERLAP_TARGETS_REACHED"
    # 三档之间也须检查协变量；不能仅因每档接近祖先就宣称严格匹配。
    xi_values = [compute_metrics(g)["xi_A"] for g in variants.values()]
    report["xi_A_matched_between_variants"] = (
        max(xi_values) - min(xi_values) <= p["xi_A_tolerance"] + 1e-12
        if all(v is not None for v in xi_values) else None
    )
    return variants, report


if __name__ == "__main__":
    from .generators import generate_graph
    from .utils import load_config

    parser = argparse.ArgumentParser(description="单祖先组的固定三元超度重连检查")
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--attempts-per-edge", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.attempts_per_edge is not None:
        config["rewiring"]["exploration_attempts_per_edge"] = args.attempts_per_edge
        config["rewiring"]["target_attempts_per_edge"] = args.attempts_per_edge
    ancestor, _ = generate_graph(config, hyperedge="Overlap", seed=args.seed)
    _, report = generate_matched_variants(ancestor, config, args.seed)
    # 命令行只打印统计；完整停留轨迹保留在函数返回值中供诊断。
    for chain in report["exploration"]["chains"]:
        chain.pop("C2_history")
    for variant in report["variants"].values():
        variant.pop("C2_history")
    print(json.dumps(report, ensure_ascii=False, indent=2))
