# 已有实验结果汇总

更新时间：2026-08-21

## 最新方法筛选：验证约束的尾部额外对抗步

这是 seed 42 的受控筛选结果，不是论文主表。两种方法均使用 Qwen2.5-3B、AG News、
10 个客户端、Dirichlet `alpha=0.1`、5 轮训练和相同的本地 embedding PGD-3
（`epsilon=0.05 relative_rms`）。最终评估使用固定且跨方法一致的 PGD-10 随机起点，
`epsilon=0.075`，每客户端 56 个样本。

| 方法 | Clean (%) ↑ | Robust (%) ↑ | Worst (%) ↑ | Bottom-tail (%) ↑ | Client Std (pp) ↓ | Pooled ASR (%) ↓ |
|---|---:|---:|---:|---:|---:|---:|
| FedPGD | 77.14 | 41.25 | 16.07 | 17.86 | 18.46 | 46.53 |
| FedRDA-tail-extra | **81.61** | **48.75** | **23.21** | **25.00** | **18.05** | **40.26** |
| 改变量 | +4.46 | +7.50 | +7.14 | +7.14 | -0.41 | -6.27 |

当前 FedRDA 先形成标准 FedPGD 更新，仅让验证鲁棒性最低的 bottom 客户端多训练
10 个 batch，再验证是否接纳这段额外增量。该结果说明方向值得继续，但必须补充 seed
43/44、完整测试集、10 轮及 7B/其他数据集实验后才能写入论文结论。

结果文件：

- FedPGD：`/data/yhpang/fedrda_deterministic_20260821/outputs/agnews_deterministic_eps075__fedpgd__seed42/summary.json`
- FedRDA：`/data/yhpang/fedrda_deterministic_20260821/outputs/agnews_tail_extra_fixed_selection_eps075__fedrda__seed42/summary.json`

## 1. 统计范围

- 数据集：AG News
- 模型：Qwen2.5-3B-Instruct + LoRA
- 客户端数：10
- 主实验异质性：Dirichlet `alpha=0.1`
- 随机种子：42、43、44
- 本表攻击：embedding PGD-20，`epsilon=0.03 relative_rms`，1 次随机重启
- 指标均为最终完整客户端测试集结果；主表报告三个随机种子的均值 ± 样本标准差

`Robust` 是客户端鲁棒准确率宏平均；`Worst` 是最差客户端鲁棒准确率；
`Bottom-tail` 是最弱 20% 客户端的平均鲁棒准确率；`Client Std` 越低表示客户端间差异越小；
`ASR` 是只在 clean 预测正确样本上计算的攻击成功率，越低越好。

## 2. AG News 主实验（alpha=0.1，三种子均值 ± 标准差）

| 方法 | Clean (%) ↑ | Robust (%) ↑ | Worst (%) ↑ | Bottom-tail (%) ↑ | Client Std (pp) ↓ | ASR (%) ↓ | 结果状态 |
|---|---:|---:|---:|---:|---:|---:|---|
| FedAvg | **93.34 ± 0.41** | 78.95 ± 1.51 | 63.60 ± 5.14 | 67.06 ± 2.98 | 10.09 ± 1.52 | 15.42 ± 1.24 | 有效 |
| CalFAT | 93.31 ± 0.38 | 90.43 ± 0.58 | 83.64 ± 2.01 | 84.56 ± 1.44 | 4.72 ± 0.55 | 3.08 ± 0.23 | 有效 |
| FedPGD | 92.93 ± 0.44 | 91.25 ± 0.46 | 85.22 ± 1.98 | 86.10 ± 1.12 | 4.36 ± 0.46 | 1.81 ± 0.08 | 有效 |
| SFAT | 93.04 ± 0.59 | **91.36 ± 0.74** | **85.26 ± 0.91** | **86.12 ± 0.93** | **4.20 ± 0.46** | 1.81 ± 0.25 | 有效 |
| FedRDA | 92.71 ± 0.55 | 91.10 ± 0.63 | 83.77 ± 2.14 | 85.18 ± 1.37 | 4.64 ± 0.66 | **1.73 ± 0.11** | 有效，但未超过关键 baseline |
| QFedAvg-EAT | 25.09 ± 5.73 | 4.50 ± 2.79 | 0.00 ± 0.00 | 0.02 ± 0.04 | 4.73 ± 3.42 | 81.92 ± 11.24 | **训练失败，不可用于论文比较** |

说明：QFedAvg-EAT 的 clean accuracy 接近 AG News 四分类的随机水平，而且三个种子均失败。
其较低的 `Client Std` 没有公平性意义，因为所有客户端的模型性能都已崩溃。

## 3. 逐种子 PGD-20 结果

### 3.1 平均鲁棒准确率

| 方法 | seed 42 | seed 43 | seed 44 |
|---|---:|---:|---:|
| FedAvg | 80.68 | 78.22 | 77.95 |
| CalFAT | 91.00 | 90.46 | 89.84 |
| FedPGD | 91.78 | 91.03 | 90.93 |
| SFAT | 91.99 | 91.57 | 90.54 |
| FedRDA | 91.83 | 90.74 | 90.74 |
| QFedAvg-EAT | 7.64 | 2.33 | 3.51 |

### 3.2 最差客户端鲁棒准确率

| 方法 | seed 42 | seed 43 | seed 44 |
|---|---:|---:|---:|
| FedAvg | 63.95 | 58.29 | 68.55 |
| CalFAT | 84.74 | 81.32 | 84.87 |
| FedPGD | 85.39 | 83.16 | 87.11 |
| SFAT | 86.32 | 84.74 | 84.74 |
| FedRDA | 84.21 | 81.45 | 85.66 |
| QFedAvg-EAT | 0.00 | 0.00 | 0.00 |

## 4. 当前结果支持的结论

### 4.1 问题现象成立

FedAvg 的 clean accuracy 为 93.34%，但 PGD-20 鲁棒准确率只有 78.95%，最差客户端只有
63.60%，客户端标准差达到 10.09 个百分点。这说明普通联邦训练同时存在平均鲁棒性不足和
客户端鲁棒性差异。

### 4.2 对抗训练有效

FedPGD、CalFAT、SFAT 和 FedRDA 都把平均鲁棒准确率提高到 90% 以上，并将客户端标准差
降低到约 4--5 个百分点。与 FedAvg 相比，FedRDA 的平均鲁棒准确率提高 12.15 个百分点，
最差客户端提高 20.18 个百分点。

### 4.3 当前 FedRDA 的独立优势尚未成立

与 SFAT 相比，FedRDA：

- 平均鲁棒准确率低 0.26 个百分点；
- 最差客户端低 1.49 个百分点；
- Bottom-tail 低 0.94 个百分点；
- 客户端标准差高 0.43 个百分点；
- clean accuracy 低 0.34 个百分点。

FedRDA 的 ASR 比 SFAT 低约 0.07 个百分点，该差异很小，当前不足以构成优势。

与 FedPGD 相比结论相同：FedRDA 的平均鲁棒性低 0.15 个百分点，最差客户端低 1.45 个
百分点，客户端标准差高 0.28 个百分点。因此，当前版本还不能支持“在保持 clean performance
的同时获得更高平均鲁棒性和更低客户端差异”的核心主张。

## 5. FedRDA 机制诊断

- 客户端鲁棒 residual 的平均两两余弦相似度接近 0；
- residual 冲突率约为 29%--38%；
- FedRDA 的约束优化能够成功求解，并能提高选中 tail 客户端的方向对齐度。

这些结果支持“客户端鲁棒更新存在冲突”这一中间动机，但方向对齐尚未转化为最终的
worst-client robustness 优势。

目前 tail 客户端按照 `clean accuracy - robust accuracy` 选择。这与论文真正关心的“鲁棒
准确率最低客户端”并不完全一致：seed 42 识别出的两个 tail 客户端与最终最弱两个客户端吻合，
seed 43 只吻合一个。后续应优先考虑用 robust loss 或 robust accuracy 定义 tail。

## 6. 攻击强度校准

使用 FedAvg seed 42、每客户端固定抽取 50 条样本进行 PGD-10 校准：

| epsilon | Clean (%) | Robust (%) | Worst (%) | Bottom-tail (%) | ASR (%) | 定位 |
|---:|---:|---:|---:|---:|---:|---|
| 0.03 | 93.0 | 82.0 | 54.0 | 62.0 | 11.8 | 弱攻击 |
| 0.05 | 93.0 | 66.2 | 30.0 | 41.0 | 28.8 | 主攻击候选 |
| 0.10 | 93.0 | 35.0 | 10.0 | 11.0 | 62.4 | 强压力测试 |

当前主实验使用 `epsilon=0.03`，鲁棒方法的准确率都集中在 90%--92%，存在天花板效应。
因此不能只依靠当前主表决定最终方法优劣。应对 FedAvg、FedPGD、SFAT 和 FedRDA 的现有
checkpoint 统一补做 `epsilon=0.05` 全量评估，并将 `epsilon=0.10` 作为压力测试。

## 7. non-IID 扩展实验（alpha=0.5）

目前仅完成 seed 42 的 FedPGD 和 SFAT；FedRDA 正在进行最终评估，因此以下结果只能作为
临时记录，不能作跨方法最终结论。

| 方法 | Clean (%) | Robust (%) | Worst (%) | Bottom-tail (%) | Client Std (pp) | ASR (%) |
|---|---:|---:|---:|---:|---:|---:|
| FedPGD | 93.91 | 92.38 | 89.21 | 89.54 | 2.49 | 1.63 |
| SFAT | 93.93 | 92.43 | 89.47 | 89.80 | 2.44 | 1.60 |

与 `alpha=0.1` 相比，`alpha=0.5` 的数据分布更加接近 IID，因此最差客户端结果更高、客户端
差异更小，趋势符合预期。

## 8. 暂不应放入论文的结果

1. QFedAvg-EAT：三个种子均未学习成功，必须修复或删除。
2. 仅一次随机重启的 PGD 结果：可用于训练期监控，但最终论文应对关键设置增加重启次数。
3. 仅 FedAvg 的 epsilon 校准：可用于选择攻击区间，不能代替所有方法的完整 epsilon 曲线。
4. 尚未完成的 alpha=0.5、alpha=1.0、DBPedia、7B 和消融实验。

## 9. 建议的下一步优先级

1. 完成当前 alpha=0.5 FedRDA seed 42 的最终评估。
2. 使用现有 checkpoint 对四个关键方法补做 `epsilon=0.05` 的全量测试。
3. 若 FedRDA 在 `epsilon=0.05` 下仍不优于 SFAT/FedPGD，先修改 tail 定义和优化目标，再继续
   大规模 DBPedia、7B 和消融实验。
4. 修复 QFedAvg-EAT 的聚合尺度，并先用 5 轮小实验确认 clean accuracy 能正常上升。
5. 最终结果报告三个随机种子的均值与标准差，并加入离散文本攻击。

## 10. 结果文件位置

- 主实验：`/data/yhpang/fedrda_runs_20260727/outputs/agnews_qwen3b/agnews_alpha01__<method>__seed<seed>/summary.json`
- non-IID：`/data/yhpang/fedrda_runs_20260727/outputs/agnews_qwen3b/agnews_alpha05__<method>__seed<seed>/summary.json`
- epsilon 校准：`/data/yhpang/fedrda_runs_20260727/epsilon_sweep/fedavg_seed42_pgd10.json`
- 运行日志：`/data/yhpang/fedrda_runs_20260727/run_logs/`
