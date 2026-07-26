# FedRDA：联邦大模型的平均、尾部与 Clean 鲁棒性

本仓库研究一个具体问题：联邦对抗训练可能提高全局模型的平均鲁棒性，却让一部分 non-IID 客户端在攻击下仍然接近失效。目标是训练一个统一的联邦大模型，使它同时具有：

- 较高的客户端宏平均对抗鲁棒性；
- 较高的 bottom-20% 和 worst-client 鲁棒性；
- 与普通联邦微调接近的 clean performance。

Embedding-space PGD 是高效的训练代理攻击，不被当作唯一的现实攻击。最终评估同时使用更强的 embedding PGD、BERT-Attack 和 DeepWordBug，检验鲁棒性能否迁移到真实离散文本。

## 方法

所有方法使用相同的预训练模型、BF16 LoRA、数据划分、通信轮数和攻击预算。

完整方法称为 **FedRDA**：

1. 客户端从同一个全局 LoRA 状态出发，分别计算 clean 更新和鲁棒更新。
2. 两者的差表示该客户端为了获得鲁棒性而额外需要的更新方向。
3. 服务器在保留平均 clean 更新的基础上，对鲁棒残差进行 tail-aware 约束聚合。
4. 聚合方向优先避免伤害当前鲁棒性最差的客户端，同时限制其偏离整体平均方向。

## 已实现的比较方法

- `fedavg`：clean LoRA + 样本量加权聚合。
- `fedpgd`：标准 embedding PGD 对抗训练 + FedAvg。
- `calfat`：依据客户端标签频数校准 logits，并使用校准后的 KL 对抗样本和交叉熵训练。
- `sfat`：按官方 SFAT 思路，在第一轮后将本地对抗损失最高的若干客户端更新乘以 slack 权重。
- `qfedavg_eat`：按照 q-FedAvg 的动态步长公式聚合 EAT 鲁棒更新。
- `fedrda`：完整方法，同时保护平均鲁棒性、尾部客户端鲁棒性和 clean performance。

CalFAT、SFAT 和 q-FedAvg 原本不是联邦大模型方法；本仓库只将其论文中的核心本地损失或服务器聚合规则适配到同一套 LoRA 参数上，适配细节会随配置和每轮指标一起保存。

## 模型与数据

正式实验：

- Qwen2.5-3B-Instruct：全部 baseline、数据集、攻击、随机种子和消融。
- Qwen2.5-7B-Instruct：核心 baseline 的规模验证。
- AG News：受控的 label-skew 实验。
- DBPedia-14：更大规模、更多类别的验证。

正式配置默认使用全部训练集和测试集。`label_skew_equal` 划分保证客户端样本数量近似相等，只改变标签分布，从而避免把客户端样本数量不足误认为鲁棒性差异。每个客户端测试集的最小样本数由配置强制检查。

## 环境

```bash
cd /path/to/fedllm-robustness
pip install -r requirements.txt
```

离散文本攻击是可选依赖：

```bash
pip install textattack
```

## 快速检查

快速检查只验证代码是否能够走通，结果不能写入论文：

```bash
python run_experiment.py \
  --config configs/smoke.yaml \
  --algorithm fedavg \
  --run-name smoke_fedavg
```

## 运行正式实验

单个方法：

```bash
python run_experiment.py \
  --config configs/agnews_qwen3b.yaml \
  --algorithm fedrda \
  --seed 42 \
  --run-name agnews_alpha01
```

断点续跑：

```bash
python run_experiment.py \
  --config configs/agnews_qwen3b.yaml \
  --algorithm fedrda \
  --seed 42 \
  --run-name agnews_alpha01 \
  --resume
```

完整命令、实验结果填写位置以及每项结果的解释见 [EXPERIMENTS.md](EXPERIMENTS.md)。

## 输出

每次运行保存在：

```text
outputs/<model-and-dataset>/<run-name>__<algorithm>__seed<seed>/
```

主要文件：

- `resolved_config.json`：完整配置；
- `data_split.json`：各客户端训练、验证和测试样本数及标签分布；
- `round_metrics.jsonl`：每轮本地、聚合和测试结果；
- `summary.json`：最终 clean、PGD、平均、bottom-20%、worst-client 和标准差；
- `final_model/`：最终 LoRA adapter 和分类头；
- `text_attack_metrics.json`：BERT-Attack/DeepWordBug 结果。

更新实验报告中的结果表：

```bash
python scripts/update_experiment_report.py \
  --outputs outputs \
  --report EXPERIMENTS.md
```

## 测试

```bash
python -m unittest discover -s tests -v
```

## 结果解释原则

- 主要平均指标使用 client-macro，而不是让大客户端主导的 sample-weighted accuracy。
- tail 指标使用 bottom-20% 和 worst-client，不以“方差更小”代替弱客户端确实得到改善。
- clean performance 必须与 FedAvg 和最强鲁棒性 baseline 同时比较。
- embedding PGD 只说明连续表示空间鲁棒性。
- 只有 BERT-Attack 和 DeepWordBug 也改善时，才能声称方法对真实文本扰动具有迁移鲁棒性。
- 所有攻击都必须针对每个最终模型重新生成，不能复用只针对 FedAvg 生成的样本。
