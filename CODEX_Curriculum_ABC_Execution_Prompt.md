# 交给 Codex 的任务：直接生成课程 A/B/C 并跑第一轮 RL

仓库：`https://github.com/yangjiaomianbao/reinforecment`
参考研究方案：`HGNN_DRL_Curriculum_ABC_Training_Plan_V1.pdf`

**这次直接实现并运行核心实验。不要再写一套很长的实施计划，不创建额外单测、benchmark、复杂验收脚本或大量说明文件。保留原 HGNN、DDQN、Discrete SAC、SIR/CRN 奖励，不改变数学定义；二元边始终不允许删除。尽量复用现有 `src/generators.py`、`src/dataset.py`、`src/ddqn.py`、`src/sac.py`、`src/evaluate.py`。**

## 1. 生成课程 A / B / C 网络

统一 `N=30, M2=60, M3=30, K=6`。每个网络保存 `graph_id, group_id, course_stage, generator_type, seed, split`，按独立骨架/家族隔离 train/validation/test。

- **课程 A**：3 个各 10 节点的社区；二元边 `54` 条社区内部 + `6` 条跨社区；三元超边 `18` 条社区内部 + `6` 条候选跨社区协同通道（优先 2:1 成员结构）+ `6` 条普通/冗余对照。随机化具体成员和通道位置，不标注“正确关键边”。
- **课程 B**：3 个社区；跨社区二元边数从 `{4,6,10}` 中取值，补足内部边使总数为 `60`；三元超边 `12` 条社区内部 + `12` 条多通道跨社区 + `6` 条高重叠/冗余，允许共享 1/2 个节点形成替代路径。
- **课程 C**：现有 `ER/WS/SBM × Uniform/Overlap/Community/Heterogeneous` 共 12 种组合，分布尽量均衡，不按 SIR 结果筛掉低收益网络。
- 数据量：A `20/8/8`、B `20/8/8`、C `40/16/32`，对应 train/validation/test，合计 `80/32/48`。

生成完就进入训练，不预先遍历全部超边找高收益标签，不安排逐图 SIR 筛选。

## 2. 直接训练：课程采样 vs 均匀采样

沿用现有 `DeletionEnv`，奖励仍然是 `F(H_t)-F(H_{t+1})`，六步结束，`gamma=1`。对每一种算法（DDQN、Discrete SAC）训练：

- **Curriculum**：三个连续训练阶段，采样 A/B/C 比例分别为 `80/20/0`、`30/50/20`、`10/20/70`。**必须使用同一模型、优化器和 Replay Buffer 连续训练**，阶段切换不得重新初始化。
- **Uniform**：使用同一批 80 张训练图，整个训练过程均匀抽图；其他设置一致。只改变训练图采样方式。
- 两组都从头训练，使用可比的随机种子与训练预算；每步都能删除存活三元超边，删除后重新编码。

提供两档运行：

| 档位 | 三阶段 Episode | train/val/test MC | 训练种子 |
|---|---|---|---|
| `quick`（这次先执行） | `60+90+150` | `32/128/512` | 每配置 1 个 |
| `full`（保留入口，暂不启动） | `200+300+500` | `64/256/2048` | 每配置 2 个 |

其他模型参数沿用当前 `config.yaml`；课程运行时覆写 `K=6`，不要破坏旧实验。验证间隔 `100` Episode。模型选择使用固定验证集的平均 `F6`。

## 3. 输出最少的结果并实际运行 quick

保留一个主入口，如：

```bash
python -m src.curriculum --profile quick --device cuda --output outputs/curriculum_abc_quick
```

上面的入口名称可按项目结构合理调整，但最终一定给出实际可执行命令，并**运行 quick 实验**。需要兼容 CPU fallback。

仅需输出：

1. `learning_curve.csv`：episode、课程阶段、算法、Curriculum/Uniform、训练奖励、验证 `F6`、SIR 调用数。
2. `results.csv`：测试图、课程类别、方法、`F0`、`F6`、`delta_F6`、六条删除超边 ID。测试方法至少包括 `Random`、`Overlap-Degree`、四个 RL 模型。
3. `summary.csv`：按课程 A/B/C 分别汇总各方法的平均 `F6` / `delta_F6`，直接比较 Curriculum 与 Uniform。
4. 必要的模型 checkpoint；一张简洁的学习曲线图（可选）。

本轮**不要**因为计算开销额外运行所有测试图上的 Static Top-6 / SIR-Greedy 穷举，也不要接入真实超图；这两项留待第一轮结果出来后再决定。

## 4. 完成后只汇报事实

展示 quick 训练是否跑完、A/B/C 各自的测试结果、Curriculum 与 Uniform 的差异、总耗时和 SIR 调用数；如有失败或性能未提升，直接报告，不改变奖励或挑选“好看”的测试图。不要新增无关文件、测试/验证脚本或重构其他模块。
