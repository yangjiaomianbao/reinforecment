"""双阶删边干预：固定图池、独立选择/评价 CRN、七阶段实验及断点续跑。"""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
from itertools import combinations, product
import json
import multiprocessing as mp
from pathlib import Path
import pickle
import sqlite3
import subprocess
import time

import numpy as np
import torch

from .evaluate import _bootstrap
from .generators import generate_graph
from .hypergraph import Graph, compute_metrics
from .sir import SIRMonteCarlo, SIRParameters, simulate_sir
from .utils import load_config

MODES = ('triad_only', 'pair_only', 'mixed')
POLICIES = ('Random', 'Static', 'Greedy')
CONDITIONS = [('main', .08, .16), ('weak_pair', .04, .16),
              ('strong_triad', .08, .32), ('weak_triad', .08, .08), ('zero_triad', .08, 0.)]
# 按生成器预先选定，覆盖三个骨架和四种三元机制；不读取任何传播收益。
REPRESENTATIVES = [('ER', 'Uniform'), ('ER', 'Overlap'), ('WS', 'Community'),
                   ('WS', 'Heterogeneous'), ('SBM', 'Overlap'), ('SBM', 'Community')]


def derived(seed, *indices):
    return int(np.random.SeedSequence([seed, *indices]).generate_state(1)[0])


def connect(path, initialize=False):
    db = sqlite3.connect(path, timeout=120)
    if initialize:
        # 主进程在启动 worker 前完成建表/WAL 切换，避免多个连接争抢初始化锁。
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('CREATE TABLE IF NOT EXISTS estimates (scope TEXT, mask BLOB, payload BLOB, calls INTEGER, seconds REAL, PRIMARY KEY(scope,mask))')
        db.execute('CREATE TABLE IF NOT EXISTS jobs (name TEXT PRIMARY KEY, payload TEXT)')
        db.commit()
    db.execute('PRAGMA synchronous=NORMAL')
    return db


class CachedMC(SIRMonteCarlo):
    """仅为现有 MC 添加落盘缓存；所有传播与 CRN 评价仍由 SIRMonteCarlo 执行。"""

    def __init__(self, graph, parameters, db, *, seed, mc):
        super().__init__(graph, parameters, seed=seed, mc=mc, batch=True)
        self.db = db
        self.scope = json.dumps([graph.graph_id, asdict(parameters), seed, mc], sort_keys=True)
        self.compute_seconds = 0.

    def estimate(self, alive_mask=None, *, alive_mask_2=None):
        m3 = np.ones(self.graph.M0, dtype=bool) if alive_mask is None else np.asarray(alive_mask)
        m2 = np.ones(self.graph.M2, dtype=bool) if alive_mask_2 is None else np.asarray(alive_mask_2)
        self.graph.alive_edge_ids(m3)
        if m2.shape != (self.graph.M2,) or m2.dtype != np.bool_:
            raise ValueError('二元掩码应为 bool [M2]')
        key = (m2.tobytes(), m3.tobytes())
        if key not in self._cache:
            packed = key[0] + key[1]
            saved = self.db.execute('SELECT payload FROM estimates WHERE scope=? AND mask=?', (self.scope, packed)).fetchone()
            if saved:
                self._cache[key] = pickle.loads(saved[0])
            else:
                start = time.perf_counter()
                result = super().estimate(m3, alive_mask_2=m2)
                if result['F'] is None:
                    raise RuntimeError('SIR NOT_CONVERGED，保留缓存以便诊断，不忽略失败轨迹')
                elapsed = time.perf_counter()-start
                self.compute_seconds += elapsed
                self.db.execute('INSERT INTO estimates VALUES (?,?,?,?,?)',
                                (self.scope, packed, pickle.dumps(result, protocol=5), self.mc, elapsed))
                self.db.commit()
        return dict(self._cache[key])


def masks(graph, actions=()):
    m2, m3 = np.ones(graph.M2, dtype=bool), np.ones(graph.M0, dtype=bool)
    for order, edge in actions:
        mask = {2: m2, 3: m3}[order]
        if not 0 <= edge < len(mask) or not mask[edge]:
            raise ValueError('非法或重复的原始边 ID')
        mask[edge] = False
    return m2, m3


def candidates(graph, mode):
    return ([(2, i) for i in range(graph.M2)] if mode != 'triad_only' else []) + \
           ([(3, i) for i in range(graph.M0)] if mode != 'pair_only' else [])


def estimate(estimator, actions=()):
    m2, m3 = masks(estimator.graph, actions)
    return estimator.estimate(m3, alive_mask_2=m2)


def reward(estimator, before, after):
    b2, b3 = masks(estimator.graph, before)
    a2, a3 = masks(estimator.graph, after)
    return estimator.paired_reward(b3, a3, before_mask_2=b2, after_mask_2=a2)['reward']


def select(estimator, mode, policy, K, rng, quota=None):
    available = candidates(estimator.graph, mode)
    start, calls = time.perf_counter(), estimator.trajectory_calls
    actions, costs = [], []
    if policy == 'Random':
        ranking = [available[i] for i in rng.permutation(len(available))]
    elif policy == 'Static':
        # 原图只打分一次；Mixed 两类交互使用同一个配对 ΔF 尺度。
        scores = {a: reward(estimator, [], [a]) for a in available}
        ranking = sorted(available, key=lambda a: (-scores[a], a))
    for step in range(K):
        legal = [a for a in available if a not in actions and
                 (quota is None or sum(b[0] == a[0] for b in actions) < quota[a[0]])]
        if not legal:
            raise ValueError('预算或配额超出候选集')
        if policy == 'Greedy':
            values = [reward(estimator, actions, actions+[a]) for a in legal]
            action = legal[int(np.argmax(values))]
        else:
            action = next(a for a in ranking if a in legal)
        actions.append(action)
        costs.append((estimator.trajectory_calls-calls, time.perf_counter()-start))
    masks(estimator.graph, actions)
    return actions, costs


def graph_record(graph, metadata, index, representative=False, small=False):
    skeleton, hyperedge = metadata['generator_type'].split('/')
    metrics = compute_metrics(graph, community_labels=metadata['community_labels'])
    return dict(graph_id=graph.graph_id, group_id=graph.group_id, N=graph.N,
                edge_index_2=graph.edge_index_2.tolist(), hyperedge_nodes_3=graph.hyperedge_nodes_3.tolist(),
                skeleton_type=skeleton, hyperedge_type=hyperedge, seed=metadata['seed'],
                index=index, representative=representative, small=small,
                metadata=metadata, metrics={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in metrics.items()})


def restore(record):
    return Graph(record['graph_id'], record['group_id'], record['N'],
                 record['edge_index_2'], record['hyperedge_nodes_3'])


def make_pool(config, seed):
    config = deepcopy(config)
    config['graph'].update(N=30, M2=60, M3=30)
    records = []
    for index, (sk, hy, replicate) in enumerate(product(('ER','WS','SBM'),
                                                         ('Uniform','Overlap','Community','Heterogeneous'), range(2))):
        graph, metadata = generate_graph(config, sk, hy, derived(seed, 100, index))
        records.append(graph_record(graph, metadata, index, replicate == 0 and (sk,hy) in REPRESENTATIVES))
    small = deepcopy(config)
    small['graph'].update(N=12, M2=12, M3=8, ws_k=2)
    for index, (sk, hy) in enumerate([('ER','Overlap'), ('SBM','Community')], 24):
        graph, metadata = generate_graph(small, sk, hy, derived(seed, 100, index))
        records.append(graph_record(graph, metadata, index, small=True))
    assert len({r['graph_id'] for r in records}) == 26
    return records


def base_row(record, parameters, condition, selection_seed, evaluation_seed):
    return {k:record[k] for k in ('graph_id','skeleton_type','hyperedge_type','seed')} | \
           asdict(parameters) | dict(condition=condition, N=record['N'], M2=len(record['edge_index_2']),
                                    M3=len(record['hyperedge_nodes_3']), selection_seed=selection_seed,
                                    evaluation_seed=evaluation_seed)


def outcome(base, estimator, mode, policy, actions, costs, selection_mc, repeat=0):
    start, calls = time.perf_counter(), estimator.trajectory_calls
    initial, final = estimate(estimator), estimate(estimator, actions)
    delta = initial['F'] - final['F']
    return base | dict(intervention_mode=mode, policy=policy, repeat=repeat, K=len(actions),
                       K2=sum(a[0]==2 for a in actions), K3=sum(a[0]==3 for a in actions),
                       F0=initial['F'], FK=final['F'], delta_FK=delta,
                       relative_reduction=delta/initial['F'] if initial['F'] else None,
                       P0=initial['P'], PK=final['P'],
                       ho_activation_fraction_0=initial['ho_activation_fraction'],
                       ho_activation_fraction_K=final['ho_activation_fraction'],
                       ho_probability_increment_K=final['ho_probability_increment_mean'],
                       selected_edges=json.dumps(actions), selection_mc=selection_mc,
                       evaluation_mc=estimator.mc, selection_sir_calls=costs[0],
                       evaluation_sir_calls=estimator.trajectory_calls-calls,
                       selection_seconds=costs[1], evaluation_seconds=time.perf_counter()-start)


def comparison(base, selection, evaluation, seed, budgets):
    rows = []
    for mode_index, mode in enumerate(MODES):
        for policy in POLICIES:
            for repeat in range(20 if policy=='Random' else 1):
                rng = np.random.default_rng(derived(seed, 33, mode_index, repeat))
                actions, costs = select(selection, mode, policy, max(budgets), rng)
                for K in budgets:
                    rows.append(outcome(base, evaluation, mode, policy, actions[:K], costs[K-1],
                                        0 if policy=='Random' else selection.mc, repeat))
    return rows


def execute_job(job, record, condition_index, settings, database):
    torch.set_num_threads(1)
    db = connect(database)
    saved = db.execute('SELECT payload FROM jobs WHERE name=?', (job,)).fetchone()
    if saved:
        db.close()
        return job, 'cached', json.loads(saved[0])
    graph = restore(record)
    condition, beta1, beta2 = CONDITIONS[condition_index]
    parameters = SIRParameters(beta1, beta2, .2, .1, settings['max_steps'])
    # 使用独立通道；评价不依赖候选个数、策略或选择次序。
    selection_seed = derived(settings['seed'], 201, record['index'], condition_index)
    evaluation_seed = derived(settings['seed'], 202, record['index'], condition_index)
    small = record['small']
    selection_mc = 128 if small else settings['selection_mc']
    evaluation_mc = 2048 if condition_index==0 else 1024
    selection = CachedMC(graph, parameters, db, seed=selection_seed, mc=selection_mc)
    evaluation = CachedMC(graph, parameters, db, seed=evaluation_seed, mc=evaluation_mc)
    base = base_row(record, parameters, condition, selection_seed, evaluation_seed)
    rng = np.random.default_rng(derived(settings['seed'], 203, record['index']))
    started = time.perf_counter()
    if job.startswith('main:') or job.startswith('sensitivity:'):
        rows = comparison(base, selection, evaluation, record['seed'], [2,4,6] if condition_index==0 else [6])
    elif job.startswith('allocation:'):
        rows = []
        for k2 in range(7):
            actions, costs = select(selection, 'mixed', 'Greedy', 6, rng, {2:k2, 3:6-k2})
            rows.append(outcome(base, evaluation, 'mixed', 'Quota-Greedy', actions, costs[-1], selection_mc) |
                        dict(quota_K2=k2, quota_K3=6-k2))
    elif job.startswith('interaction:'):
        interaction_seed = derived(settings['seed'], 204, record['index'])
        interaction = CachedMC(graph, parameters, db, seed=interaction_seed, mc=256)
        rows = []
        pairs, triads = candidates(graph,'pair_only'), candidates(graph,'triad_only')
        for kind, choices in [('Pair-Pair',list(combinations(pairs,2))),
                              ('Triad-Triad',list(combinations(triads,2))), ('Pair-Triad',list(product(pairs,triads)))]:
            for index in rng.choice(len(choices),20,replace=False):
                a,b = choices[index]
                start, calls = time.perf_counter(), interaction.trajectory_calls
                baseline, one, two, both = [estimate(interaction,x) for x in ([],[a],[b],[a,b])]
                G1, G2, G12 = [baseline['F']-r['F'] for r in (one,two,both)]
                # J = -F_ij + F_i + F_j - F_0；同一随机场估计其标准误，保留正负值。
                # F 样本本身是感染人数/N；先还原整数人数，避免相消误差把严格零 J 变成伪非零。
                counts = [np.rint(r['F_samples']*graph.N).astype(np.int64)
                          for r in (baseline,one,two,both)]
                samples = (counts[1]+counts[2]-counts[3]-counts[0])/graph.N
                interaction_gain = float(samples.mean())
                table={2:graph.edge_index_2,3:graph.hyperedge_nodes_3}
                shared=len(set(table[a[0]][a[1]]) & set(table[b[0]][b[1]]))
                se=float(samples.std(ddof=1)/np.sqrt(interaction.mc))
                rows.append(base | dict(evaluation_seed=interaction_seed, interaction_type=kind,
                    edge_i=json.dumps(a),edge_j=json.dumps(b), shared_nodes=shared,shares_nodes=bool(shared),
                    F0=baseline['F'], F_i=one['F'], F_j=two['F'], F_ij=both['F'],
                    gain_i=G1,gain_j=G2,joint_gain=G12,J=interaction_gain,J_standard_error=se,
                    J_mc_ci_low=interaction_gain-1.96*se,J_mc_ci_high=interaction_gain+1.96*se,
                    evaluation_mc=256,evaluation_sir_calls=interaction.trajectory_calls-calls,
                    evaluation_seconds=time.perf_counter()-start))
    elif job.startswith('exact:'):
        rows=[]
        for mode in MODES:
            selected={}
            for policy in ('Static','Greedy'):
                selected[policy]=select(selection,mode,policy,3,rng)
            start,calls=time.perf_counter(),selection.trajectory_calls
            best_actions,best_value,count=None,float('inf'),0
            for actions in combinations(candidates(graph,mode),3):
                value=estimate(selection,actions)['F'];count+=1
                if value<best_value:
                    best_value,best_actions=value,list(actions)
            selected['Enumeration']=(best_actions,[(selection.trajectory_calls-calls,time.perf_counter()-start)])
            tested={policy:outcome(base,evaluation,mode,policy,actions,costs[-1],128)
                    for policy,(actions,costs) in selected.items()}
            baseline=estimate(selection)['F']
            for policy,(actions,_) in selected.items():
                row=tested[policy]
                row.update(search_FK=estimate(selection,actions)['F'],search_F0=baseline,
                           search_best_FK=best_value,search_gap=estimate(selection,actions)['F']-best_value,
                           independent_FK_gap_to_enumeration=row['FK']-tested['Enumeration']['FK'],
                           enumeration_sets=count,optimality='finite_MC_search_only')
                rows.append(row)
    else:
        raise ValueError(job)
    db.execute('INSERT INTO jobs VALUES (?,?)',(job,json.dumps(rows)))
    db.commit();db.close()
    return job, f'{time.perf_counter()-started:.1f}s', rows


def core_checks(records, settings):
    graph=restore(records[0]);p=SIRParameters(.08,.16,.2,.1,settings['max_steps'])
    ones2,ones3=masks(graph);none2,none3=~ones2,~ones3
    for i in range(8):
        old=simulate_sir(graph,p,seed=9001,trajectory_id=i,return_history=True)
        explicit=simulate_sir(graph,p,ones3,alive_mask_2=ones2,seed=9001,trajectory_id=i,return_history=True)
        assert np.array_equal(old['history'],explicit['history'])
        zero=SIRParameters(.08,0.,.2,.1,settings['max_steps'])
        first=simulate_sir(graph,zero,ones3,seed=9001,trajectory_id=i,return_history=True)
        second=simulate_sir(graph,zero,none3,seed=9001,trajectory_id=i,return_history=True)
        assert np.array_equal(first['history'],second['history'])
    for m2,m3 in [(ones2,ones3),(none2,ones3),(ones2,none3),(none2,none3),masks(graph,[(2,0),(3,0)])]:
        scalar=SIRMonteCarlo(graph,p,seed=9002,mc=8).estimate(m3,alive_mask_2=m2)
        estimator=SIRMonteCarlo(graph,p,seed=9002,mc=8,batch=True)
        batch=estimator.estimate(m3,alive_mask_2=m2)
        assert scalar['status']==batch['status']=='CONVERGED'
        assert np.array_equal(scalar['F_samples'],batch['F_samples'])
        assert np.array_equal(scalar['P_samples'],batch['P_samples'])
        again=estimator.estimate(m3,alive_mask_2=m2)
        assert estimator.trajectory_calls==8 and batch['F']==again['F']
    estimator=SIRMonteCarlo(graph,p,seed=9003,mc=8,batch=True)
    estimator.estimate(ones3,alive_mask_2=ones2);estimator.estimate(ones3,alive_mask_2=none2)
    assert len(estimator._cache)==2
    a2,a3=masks(graph,[(2,0)]);b2,b3=masks(graph,[(3,0)])
    assert a3.all() and b2.all() and not a2[0] and not b3[0]
    try:
        masks(graph,[(2,0),(2,0)])
    except ValueError:
        pass
    else:
        raise AssertionError('重复动作未被拒绝')
    print('A CHECKS PASSED: legacy/default masks, zero-beta2 paths, all-layer removals, CRN, batch parity, independent masks/actions.',flush=True)


def write_csv(path, rows):
    fields=list(dict.fromkeys(key for row in rows for key in row))
    temporary=path.with_suffix('.csv.tmp')
    with temporary.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
    temporary.replace(path)


def summarize(rows, seed):
    """Random 20 次先在图内平均；配对比较和 Bootstrap 的样本单位始终为图。"""
    per_graph=defaultdict(list)
    for r in rows:
        per_graph[(r['graph_id'],r['intervention_mode'],r['policy'],r['K'])].append(r)
    averages={}
    for key,records in per_graph.items():
        row=records[0].copy()
        for field in ('F0','FK','delta_FK','PK','K2','K3'):
            row[field]=float(np.mean([r[field] for r in records]))
        row['random_repeat_sd']=float(np.std([r['delta_FK'] for r in records],ddof=1)) if len(records)>1 else 0.
        averages[key]=row
    grouped=defaultdict(list)
    for r in averages.values():
        for scope,sk,hy in [('all','ALL','ALL'),('skeleton',r['skeleton_type'],'ALL'),
                            ('hyperedge','ALL',r['hyperedge_type']),('combination',r['skeleton_type'],r['hyperedge_type'])]:
            grouped[(scope,sk,hy,r['intervention_mode'],r['policy'],r['K'])].append(r)
    rng=np.random.default_rng(derived(seed,301));result=[]
    for (scope,sk,hy,mode,policy,K),records in sorted(grouped.items()):
        row=dict(scope=scope,skeleton_type=sk,hyperedge_type=hy,intervention_mode=mode,policy=policy,K=K,graph_count=len(records))
        for field in ('F0','FK','delta_FK','PK','K2','K3'):
            values=[r[field] for r in records]
            row[field]=float(np.mean(values))
            row[field+'_ci_low'],row[field+'_ci_high']=_bootstrap(values,rng,2000)
        row['mean_within_graph_repeat_sd']=float(np.mean([r['random_repeat_sd'] for r in records]))
        for reference in POLICIES:
            values=[r['delta_FK']-averages[(r['graph_id'],mode,reference,K)]['delta_FK'] for r in records]
            name='gain_vs_'+reference.lower()
            row[name]=float(np.mean(values))
            row[name+'_ci_low'],row[name+'_ci_high']=_bootstrap(values,rng,2000)
        result.append(row)
    return result


def collect(db, prefix):
    return [row for (payload,) in db.execute('SELECT payload FROM jobs WHERE name LIKE ? ORDER BY name',(prefix+':%',)) for row in json.loads(payload)]


def export(output, seed, representatives):
    db=connect(output/'progress.sqlite')
    main=collect(db,'main')
    if main:
        write_csv(output/'results.csv',main)
        write_csv(output/'summary.csv',summarize(main,seed))
    for prefix,name in [('allocation','allocation'),('interaction','interaction'),('exact','exact_search')]:
        rows=collect(db,prefix)
        if rows:write_csv(output/(name+'.csv'),rows)
    sensitivity=collect(db,'sensitivity')
    if sensitivity:
        # 主条件直接引用已完成的高精度结果，六张先导图不重复计入主实验。
        sensitivity += [r for r in main if r['graph_id'] in representatives and r['K']==6]
        write_csv(output/'parameter_sensitivity.csv',sensitivity)
    db.close()


def plots(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    def read(name):
        with (output/name).open() as f:return list(csv.DictReader(f))
    def avg(rows,field):return float(np.mean([float(r[field]) for r in rows]))
    colors=dict(triad_only='#3977b6',pair_only='#d97927',mixed='#368951')
    labels=dict(triad_only='Triad-Only',pair_only='Pair-Only',mixed='Mixed')
    styles=dict(Random=':',Static='--',Greedy='-')
    summary=read('summary.csv')
    fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
    for mode,policy in product(MODES,POLICIES):
        rows=sorted([r for r in summary if r['scope']=='all' and r['intervention_mode']==mode and r['policy']==policy],key=lambda r:int(r['K']))
        ax.plot([2,4,6],[float(r['delta_FK']) for r in rows],color=colors[mode],ls=styles[policy],marker='o',label=labels[mode]+' / '+policy)
    search_mc=max(int(r['selection_mc']) for r in read('results.csv'))
    ax.set(xlabel='Deletion budget K',ylabel='Mean reduction in final infected fraction',xticks=[2,4,6],title=f'24 fixed graphs | selection MC={search_mc}, evaluation MC=2048')
    ax.legend(ncol=3,fontsize=8);ax.grid(alpha=.2)
    fig.savefig(output/'figure_1_intervention_comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.8),sharey=True,layout='constrained')
    combos=[(s,h) for s,h in product(('ER','WS','SBM'),('Uniform','Overlap','Community','Heterogeneous'))]
    for ax,mode in zip(axes,MODES):
        for policy in ('Random','Greedy'):
            vals=[]
            for sk,hy in combos:
                r=next(r for r in summary if r['scope']=='combination' and r['skeleton_type']==sk and r['hyperedge_type']==hy and r['intervention_mode']==mode and r['policy']==policy and r['K']=='6')
                vals.append(float(r['delta_FK']))
            ax.plot(range(12),vals,ls=styles[policy],marker='o',label=policy,color=colors[mode])
        ax.set(title=labels[mode],xticks=range(12),xticklabels=[s+'/'+h[:3] for s,h in combos]);ax.tick_params(axis='x',rotation=65,labelsize=8);ax.legend();ax.grid(alpha=.2)
    axes[0].set_ylabel('Mean reduction at K=6 (2 graphs per combination)')
    fig.savefig(output/'figure_2_structure_comparison.png',dpi=180);plt.close(fig)
    rows=read('allocation.csv');fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
    for graph in sorted({r['graph_id'] for r in rows}):
        group=sorted([r for r in rows if r['graph_id']==graph],key=lambda r:int(r['K2']))
        ax.plot(range(7),[float(r['delta_FK']) for r in group],alpha=.55,lw=1,label=group[0]['skeleton_type']+'/'+group[0]['hyperedge_type'])
    ax.plot(range(7),[avg([r for r in rows if int(r['K2'])==k],'delta_FK') for k in range(7)],color='black',lw=2.5,marker='o',label='Graph mean')
    ax.set(xlabel='Fixed quota (K2, K3)',ylabel='Reduction in final infected fraction',title='Quota-constrained Greedy | six prespecified graphs',xticks=range(7),xticklabels=[f'({k},{6-k})' for k in range(7)])
    ax.legend(fontsize=8,ncol=2);ax.grid(alpha=.2)
    fig.savefig(output/'figure_3_mixed_allocation.png',dpi=180);plt.close(fig)
    rows=read('interaction.csv');fig,ax=plt.subplots(figsize=(8,5),layout='constrained');kinds=['Pair-Pair','Triad-Triad','Pair-Triad']
    values=[[float(r['J']) for r in rows if r['interaction_type']==kind] for kind in kinds]
    ax.boxplot(values,labels=kinds,showfliers=False)
    rng=np.random.default_rng(7)
    for i,v in enumerate(values,1):ax.scatter(i+rng.uniform(-.15,.15,len(v)),v,alpha=.3,s=12)
    ax.axhline(0,color='black',ls='--',lw=1);ax.set(ylabel='J = joint gain - sum of individual gains',title='20 pairs/type/graph | six graphs | paired MC=256')
    fig.savefig(output/'figure_4_interaction.png',dpi=180);plt.close(fig)
    rows=read('parameter_sensitivity.csv');fig,axes=plt.subplots(1,3,figsize=(14,4.7),sharey=True,layout='constrained')
    for ax,policy in zip(axes,POLICIES):
        for mode in MODES:
            means=[]
            for name,_,_ in CONDITIONS:
                subset=[r for r in rows if r['condition']==name and r['policy']==policy and r['intervention_mode']==mode]
                means.append(avg(subset,'delta_FK'))
            ax.plot(range(5),means,color=colors[mode],marker='o',label=labels[mode])
        ax.set(title=policy,xticks=range(5),xticklabels=[f'{b1:.2f}/{b2:.2f}' for _,b1,b2 in CONDITIONS],xlabel='beta1 / beta2');ax.tick_params(axis='x',rotation=35);ax.legend(fontsize=8);ax.grid(alpha=.2)
    axes[0].set_ylabel('Mean reduction at K=6 | six fixed graphs')
    fig.savefig(output/'figure_5_parameter_sensitivity.png',dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config.yaml')
    parser.add_argument('--output',default='outputs/mixed_intervention_v1')
    parser.add_argument('--run-all',action='store_true',required=True)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--selection-mc',type=int,choices=[32,64],default=64)
    args=parser.parse_args()
    torch.set_num_threads(1)
    config=load_config(args.config);output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    settings=dict(seed=config['seed'],selection_mc=args.selection_mc,random_repeats=20,
                  evaluation_mc=2048,interaction_mc=256,exact_selection_mc=128,
                  sensitivity_mc=1024,max_steps=2000,budgets=[2,4,6])
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path('src').glob('*.py'))}
    manifest_path=output/'run_config.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if manifest['settings']!=settings or manifest['source_sha256']!=hashes or manifest['config']!=config:
            raise ValueError('续跑配置/代码不同，请明确使用新的输出目录，避免混合实验')
        records=manifest['graphs']
    else:
        import numba
        records=make_pool(config,settings['seed'])
        manifest=dict(settings=settings,config=config,conditions=CONDITIONS,graphs=records,source_sha256=hashes,
                      git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                      git_status=subprocess.check_output(['git','status','--short'],text=True),
                      started_utc=datetime.now(timezone.utc).isoformat(),numpy_version=np.__version__,
                      numba_version=numba.__version__,workers=args.workers,
                      random_repeat_unit='20 sets per graph/mode, average within graph before bootstrap',
                      resource_scope='selection costs cumulative within a sequence; evaluation costs incremental with shared persistent cache; actual unique computations in progress.sqlite',
                      tie_break='order then original edge ID, no gain-based graph filtering',
                      mc_random_field='seed / channel 2 / episode 0 / trajectory -> initial,infection,recovery; 64-step blocks',
                      original_compatibility='Before change vs shared kernel: 32 complete histories identical; MC64 F/P samples identical (checked during implementation).')
        manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    connect(output/'progress.sqlite',initialize=True).close()
    core_checks(records,settings)
    representative=[r for r in records if r['representative']]
    ids={r['graph_id'] for r in representative}
    phases=[('B pilot',[(f"main:{r['index']}",r,0) for r in representative]),
            ('C full',[(f"main:{r['index']}",r,0) for r in records if not r['small'] and not r['representative']]),
            ('D allocation',[(f"allocation:{r['index']}",r,0) for r in representative]),
            ('E interaction',[(f"interaction:{r['index']}",r,0) for r in representative]),
            ('F exact',[(f"exact:{r['index']}",r,0) for r in records if r['small']]),
            ('G sensitivity',[(f"sensitivity:{r['index']}:{c}",r,c) for c in range(1,5) for r in representative])]
    started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn')) as executor:
        for phase,jobs in phases:
            print('START',phase,'graphs/conditions',len(jobs),flush=True)
            db=connect(output/'progress.sqlite')
            completed={name for (name,) in db.execute('SELECT name FROM jobs')}
            db.close()
            pending=[(job,r,c) for job,r,c in jobs if job not in completed]
            print('REUSE',phase,len(jobs)-len(pending),'completed jobs',flush=True)
            futures=[executor.submit(execute_job,job,r,c,settings,str(output/'progress.sqlite')) for job,r,c in pending]
            for future in as_completed(futures):
                job,duration,rows=future.result()
                print('DONE',phase,job,duration,'rows',len(rows),flush=True)
                export(output,settings['seed'],ids)
            print('FINISHED',phase,flush=True)
    export(output,settings['seed'],ids);plots(output)
    db=connect(output/'progress.sqlite')
    calls,seconds,estimates=db.execute('SELECT SUM(calls),SUM(seconds),COUNT(*) FROM estimates').fetchone()
    manifest.update(completed_utc=datetime.now(timezone.utc).isoformat(),last_run_wall_seconds=time.perf_counter()-started,
                    unique_sir_trajectories=calls,unique_estimates=estimates,sum_estimate_seconds=seconds,
                    completed_jobs=db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0])
    db.close();manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    with (output/'summary.csv').open() as f:
        summary=[r for r in csv.DictReader(f) if r['scope']=='all' and r['K']=='6']
    print('COMPLETE',json.dumps({'wall_seconds':manifest['last_run_wall_seconds'],'unique_sir_trajectories':calls,
                               'output':str(output),'K6':[{k:r[k] for k in ('intervention_mode','policy','delta_FK','gain_vs_random','gain_vs_static')} for r in summary]}),flush=True)


if __name__=='__main__':
    main()
