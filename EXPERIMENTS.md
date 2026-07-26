# 实验运行与结果解读手册

这份文件写给第一次接触科研实验的同学。请按照顺序执行，不要跳过 smoke test，也不要把 smoke test 或失败运行的数字写进论文。

本项目研究的问题是：联邦对抗训练可能具有较高的平均鲁棒性，但一部分 non-IID 客户端在攻击下仍然接近失效。实验需要同时检查平均鲁棒性、尾部客户端鲁棒性和 clean performance。

---

## 1. 开始前需要理解的几个概念

### 1.1 一次 run 是什么

一次 run 指“一个模型、一个数据集、一个 non-IID 设置、一个方法、一个随机种子”的完整训练。

例如：

```text
Qwen2.5-3B + AG News + alpha=0.1 + FedRDA + seed=42
```

这是一个 run。把 seed 换成 43，就是另一个独立 run。

### 1.2 为什么要运行三个随机种子

客户端数据划分、模型初始化和 batch 顺序存在随机性。单个 seed 的结果可能只是偶然现象。正式实验使用：

```text
42、43、44
```

论文最终报告三个种子的平均值和标准差，不能只挑结果最好的 seed。

### 1.3 smoke test 和正式实验的区别

smoke test 只使用很少的数据和一个通信轮次，用来检查代码、显存和输出文件是否正常。它不具有统计意义。

正式实验使用完整训练集、完整测试集和完整通信轮数。只有正式实验可以进入论文。

---

## 2. 进入项目目录

服务器连接方式已经单独提供，本手册不重复服务器名称或登录命令。登录后进入
你实际存放本仓库的目录，例如：

```bash
cd /path/to/fedllm-robustness
pwd
```

将 `/path/to` 换成实际路径。`pwd` 输出的最后一级目录应当是
`fedllm-robustness`。

查看项目文件：

```bash
ls
```

至少应看到：

```text
README.md
EXPERIMENTS.md
run_experiment.py
evaluate_text_attacks.py
configs/
fedrda_experiments/
scripts/
tests/
```

---

## 3. 检查 Python 环境

先激活已经为本项目准备好的 Python 环境，然后设置：

```bash
export PYTHON_BIN=python
```

如果还没有环境，可以创建一个独立环境：

```bash
conda create -n fedrda python=3.12 -y
conda activate fedrda
export PYTHON_BIN=python
```

先按照所在服务器的 CUDA 版本安装对应的 PyTorch。不同服务器的 CUDA 驱动可能
不同，所以本手册不提供固定的 PyTorch 安装命令。确认 PyTorch 可用后安装其余
依赖：

```bash
$PYTHON_BIN -m pip install -r requirements.txt
```

检查 Python：

```bash
$PYTHON_BIN --version
```

检查核心依赖：

```bash
$PYTHON_BIN -c "import torch, transformers, peft, datasets, scipy; print('environment ok')"
```

如果输出：

```text
environment ok
```

说明环境可用。

如果出现 `ModuleNotFoundError`，先执行：

```bash
$PYTHON_BIN --version
$PYTHON_BIN -c "import torch, transformers, peft, datasets, scipy; print('environment ok')"
```

把完整报错和上述输出发给项目负责人。不要直接运行
`pip install -U torch`，否则可能把服务器原本匹配的 PyTorch/CUDA 组合破坏。
`requirements.txt` 不安装 PyTorch，PyTorch 必须根据实际 CUDA 环境单独安装。

离散文本攻击另外需要 TextAttack。安装前先检查：

```bash
$PYTHON_BIN -c "import textattack; print('textattack ok')"
```

如果这里报错，请让项目负责人安装，不要在正式实验环境中直接执行大规模
`pip install -U`。不要在不同实验中随意更换 PyTorch、Transformers 或 PEFT
版本；环境变化必须记录。

---

## 4. 检查 GPU

运行：

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader
```

选择显存占用较低、利用率低、没有其他训练进程的 GPU。若所有卡的利用率都很高，
先等待，不要为了赶进度挤到别人的训练进程上。假设选择 GPU 0：

```bash
export CUDA_VISIBLE_DEVICES=0
```

再次检查：

```bash
$PYTHON_BIN -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

应当输出 `True` 和 GPU 名称。

注意：设置 `CUDA_VISIBLE_DEVICES=3` 后，程序内部看到的仍然是 `cuda:0`，这是正常现象。

---

## 5. 先运行单元测试

```bash
$PYTHON_BIN -m unittest discover -s tests -v
```

单元测试检查：

- equal-size label-skew 是否使用每一个样本且没有重复；
- 每个客户端样本量是否近似相等；
- SFAT 是否提高高损失客户端权重；
- q-FedAvg 动态步长是否正确；
- tail-aware 约束聚合是否满足方向约束。

只有全部显示 `ok`，最后出现 `OK`，才能继续。测试失败时不要启动正式训练。

---

## 6. 运行 smoke test

### 6.1 逐个检查所有方法

```bash
for method in fedavg fedpgd calfat sfat qfedavg_eat fedrda; do
  $PYTHON_BIN run_experiment.py \
    --config configs/smoke.yaml \
    --algorithm "$method" \
    --seed 42 \
    --run-name "smoke_${method}"
done
```

正常完成时，终端最后会出现：

```text
completed: outputs/smoke/...
```

### 6.2 smoke test 检查什么

检查输出目录：

```bash
find outputs/smoke -maxdepth 2 -type f | sort
```

每个方法至少应生成：

```text
resolved_config.json
data_split.json
round_metrics.jsonl
latest_checkpoint.pt
summary.json
final_model/
```

检查是否有错误：

```bash
grep -R "Traceback\|CUDA out of memory\|nan" outputs/smoke
```

没有输出通常表示没有发现这些错误。

### 6.3 smoke 结果不能说明什么

smoke test 不能证明：

- 方法优于 baseline；
- 客户端鲁棒性问题真实存在；
- 3B 或 7B 模型有效；
- 方法能够迁移到真实文本攻击。

它只证明程序能够运行。

---

## 7. 配置文件怎么看

以 `configs/agnews_qwen3b.yaml` 为例。

### data 部分

- `max_train_samples: null`：使用全部训练集。
- `max_test_samples: null`：使用全部测试集。
- `num_clients: 10`：十个客户端。
- `dirichlet_alpha: 0.1`：强 label-skew。
- `partition_mode: label_skew_equal`：客户端总样本数接近相等，只让标签分布不同。
- `min_client_test_samples: 100`：测试样本不足时程序直接报错，不允许继续产生不可信结果。

### model 部分

- `name_or_path`：默认是 Hugging Face 模型名。服务器能够联网时会自动下载；
  如果模型权重已经存放在本地，把它改成该服务器上的实际绝对路径。
- `lora_rank`：LoRA rank。
- `target_modules`：插入 LoRA 的注意力模块。
- `dtype: bfloat16`：使用 BF16。

### attack 部分

- `epsilon`：embedding 扰动强度。
- `train_steps: 3`：训练使用三步 PGD。
- `eval_steps: [10, 20, 50]`：测试使用更强攻击。
- `eval_restarts: 5`：最终强攻击使用多次随机初始化。

### federated 部分

- `rounds: 50`：通信轮数。
- `algorithm`：默认方法，可被命令行覆盖。
- `adv_weight`：本地鲁棒目标中的对抗损失权重。
- `clean_consistency_weight`：clean—adversarial 一致性权重。
- `tail_ratio: 0.2`：最差 20% 客户端视为 tail clients。
- `evaluate_every: 5`：每五轮完整评估一次。

每次运行都会复制实际使用的完整配置到 `resolved_config.json`。论文参数以该文件为准，而不是依靠记忆。

---

## 8. 主实验一：Qwen2.5-3B + AG News

### 8.1 每个方法解决什么

- `fedavg`：无对抗防御。
- `fedpgd`：标准 embedding PGD 联邦对抗训练。
- `calfat`：处理 label-skew 的校准对抗训练。
- `sfat`：处理对抗训练加剧的客户端异构性。
- `qfedavg_eat`：q-FedAvg 客户端公平优化与 EAT 鲁棒目标的组合。
- `fedrda`：完整方法，同时优化平均鲁棒性、尾部客户端鲁棒性和 clean performance。

### 8.2 推荐的后台运行方式

先创建日志目录：

```bash
mkdir -p run_logs
```

运行一个方法：

```bash
nohup env CUDA_VISIBLE_DEVICES=0 $PYTHON_BIN -u run_experiment.py \
  --config configs/agnews_qwen3b.yaml \
  --algorithm fedavg \
  --seed 42 \
  --run-name agnews_alpha01 \
  > run_logs/agnews_alpha01__fedavg__seed42.log 2>&1 &
```

终端会返回一个 PID。保存 PID：

```bash
echo $! > run_logs/agnews_alpha01__fedavg__seed42.pid
```

查看进度：

```bash
tail -f run_logs/agnews_alpha01__fedavg__seed42.log
```

退出 `tail -f` 使用 `Ctrl+C`，不会停止训练。

检查进程：

```bash
ps -fp "$(cat run_logs/agnews_alpha01__fedavg__seed42.pid)"
```

### 8.3 自动顺序运行全部方法

同一张 GPU 上不要并行塞入所有方法。使用脚本顺序执行：

```bash
export PYTHON_BIN=python
export SEEDS="42 43 44"
export METHODS="fedavg fedpgd calfat sfat qfedavg_eat fedrda"
export SUITE_NAME="agnews_alpha01"

nohup bash scripts/run_suite.sh configs/agnews_qwen3b.yaml 0 \
  > run_logs/agnews_qwen3b_suite.log 2>&1 &

echo $! > run_logs/agnews_qwen3b_suite.pid
```

查看总进度：

```bash
tail -f run_logs/agnews_qwen3b_suite.log
```

### 8.4 如何判断完成

日志中每个 run 应出现：

```text
completed: outputs/agnews_qwen3b/...
```

统计已经完成多少个 run：

```bash
find outputs/agnews_qwen3b -name summary.json | wc -l
```

七个方法乘三个种子，完整主实验应有 21 个 `summary.json`。

### 8.5 这个实验回答什么

1. 普通 FedPGD 在提高平均鲁棒性时是否损害 clean performance？
2. 现有 baseline 的平均鲁棒性提高后，是否仍有部分客户端接近失效？
3. CalFAT、SFAT 和 q-FedAvg-EAT 能否解决尾部客户端问题？
4. 完整方法是否在不降低平均鲁棒性的情况下提高 bottom-20% 和 worst-client？

---

## 9. 主实验二：non-IID 强度

使用相同数据、模型和训练参数，只改变 `dirichlet_alpha`：

- `1.0`：较弱异构；
- `0.5`：中等异构；
- `0.1`：强异构。

命令行可以直接覆盖配置，不需要复制 YAML：

```bash
$PYTHON_BIN run_experiment.py \
  --config configs/agnews_qwen3b.yaml \
  --algorithm fedrda \
  --seed 42 \
  --dirichlet-alpha 1.0 \
  --run-name agnews_alpha10
```

至少比较：

```text
fedpgd、sfat、fedrda
```

每个 alpha 使用相同的 seed 42、43、44。

这个实验回答：

- non-IID 越强时，tail/worst robustness 是否恶化；
- 客户端鲁棒残差的方向冲突是否增加；
- 完整方法是否主要在强异构情况下有效。

注意：run name 必须包含 alpha，防止不同实验写入同一目录。

---

## 10. 主实验三：DBPedia-14

运行完整 baseline：

```bash
export SEEDS="42 43 44"
export METHODS="fedavg fedpgd calfat sfat qfedavg_eat fedrda"
export SUITE_NAME="dbpedia_alpha01"

nohup bash scripts/run_suite.sh configs/dbpedia_qwen3b.yaml 0 \
  > run_logs/dbpedia_qwen3b_suite.log 2>&1 &
```

这个实验回答：

- 结果是否只是 AG News 四分类的特殊现象；
- 类别更多时客户端鲁棒性保护不足是否仍存在；
- CalFAT 在多类别场景是否比 tail-aware 聚合更合适。

如果方法只在 AG News 有效，就不能在论文中声称对一般文本分类任务都有效。

---

## 11. 主实验四：Qwen2.5-7B

7B 运行核心方法：

```bash
export SEEDS="42 43 44"
export METHODS="fedavg fedpgd sfat qfedavg_eat fedrda"
export SUITE_NAME="agnews_alpha01"

nohup bash scripts/run_suite.sh configs/agnews_qwen7b.yaml 0 \
  > run_logs/agnews_qwen7b_suite.log 2>&1 &
```

这个实验回答：

- 方法是否能够从 3B 扩展到 7B；
- 模型容量增大后客户端尾部失效是否自动消失；
- 对更大模型的 LoRA 更新进行约束聚合是否稳定。

7B 发生 OOM 时，优先降低 `batch_size` 和 `eval_batch_size`，不要先减少客户端测试样本。

---

## 12. 离散文本攻击

### 12.1 为什么还要运行离散攻击

embedding PGD 是连续空间代理攻击。真实用户输入的是 token，因此必须使用真实可读文本攻击检验迁移性：

- BERT-Attack：词级语义替换；
- DeepWordBug：字符插入、删除、交换等。

### 12.2 先做攻击流水线检查

下面的 20 条/客户端只能检查程序，不能写进论文：

```bash
$PYTHON_BIN evaluate_text_attacks.py \
  --config configs/agnews_qwen3b.yaml \
  --run-dir outputs/agnews_qwen3b/agnews_alpha01__fedrda__seed42 \
  --attacks bert_attack deepwordbug \
  --num-examples-per-client 20 \
  --query-budget 1000 \
  --seed 42
```

### 12.3 正式离散攻击

正式命令不限制客户端测试样本：

```bash
$PYTHON_BIN evaluate_text_attacks.py \
  --config configs/agnews_qwen3b.yaml \
  --run-dir outputs/agnews_qwen3b/agnews_alpha01__fedrda__seed42 \
  --attacks bert_attack deepwordbug \
  --num-examples-per-client -1 \
  --query-budget 1000 \
  --seed 42
```

至少对以下最终模型分别运行：

```text
FedAvg、FedPGD、CalFAT、SFAT、q-FedAvg-EAT、FedRDA
```

每个模型必须重新生成攻击样本。不能只攻击 FedAvg，再把相同样本拿去测试所有方法。

输出文件：

```text
<run-dir>/text_attack_metrics.json
```

---

## 13. 消融实验

消融实验只在 Qwen2.5-3B + AG News 上进行，使用 seed 42、43、44。

### 13.1 去掉 clean consistency

```bash
$PYTHON_BIN run_experiment.py \
  --config configs/agnews_qwen3b.yaml \
  --algorithm fedrda \
  --seed 42 \
  --clean-consistency-weight 0 \
  --run-name ablation_no_consistency
```

检查 clean performance 是否明显下降。

### 13.2 去掉 tail 保护

将 warmup 覆盖完整训练轮数目前需要复制一份配置，把：

```yaml
warmup_rounds: 50
```

此时服务器始终使用平均鲁棒残差，不执行 tail 约束。run name 使用：

```text
ablation_no_tail
```

检查 bottom-20% 和 worst-client 提升是否消失。

### 13.3 改变 tail ratio

```bash
--tail-ratio 0.1
--tail-ratio 0.2
--tail-ratio 0.3
```

检查方法是否依赖某个特殊比例。

### 13.4 改变 LoRA rank

```bash
--lora-rank 4
--lora-rank 8
--lora-rank 16
```

检查性能、显存和通信参数量之间的关系。

### 13.5 改变 residual weight

```bash
--residual-weight 0.5
--residual-weight 1.0
--residual-weight 1.5
```

检查尾部提升是否只是因为全局更新幅度更大。

---

## 14. 断点续跑

如果训练因服务器重启或 SSH 断开而停止，使用与原 run 完全相同的参数并增加：

```bash
--resume
```

例如：

```bash
$PYTHON_BIN run_experiment.py \
  --config configs/agnews_qwen3b.yaml \
  --algorithm fedrda \
  --seed 42 \
  --run-name agnews_alpha01 \
  --resume
```

不要改变 config、algorithm、seed 或 run name，否则可能续跑错误实验。

如果没有 `latest_checkpoint.pt`，不能续跑。

---

## 15. 结果文件怎么看

一个完整 run 的目录结构如下：

```text
outputs/agnews_qwen3b/agnews_alpha01__fedrda__seed42/
├── resolved_config.json
├── data_split.json
├── round_metrics.jsonl
├── latest_checkpoint.pt
├── summary.json
├── text_attack_metrics.json
└── final_model/
```

### resolved_config.json

记录实际使用的全部参数。首先确认：

- 模型路径正确；
- seed 正确；
- algorithm 正确；
- max train/test samples 为 null；
- num clients 为 10；
- alpha 与 run name 一致。

### data_split.json

检查每个客户端：

- train/test 样本数是否充足；
- 不同客户端总样本数是否接近；
- 标签分布是否确实不同；
- 是否存在测试集为空或极小的客户端。

### round_metrics.jsonl

每一行是一轮训练。可以查看：

- 每轮选中了哪些客户端；
- 本地损失；
- tail client IDs；
- 残差方向冲突；
- 聚合是否进入 fallback；
- 每轮 clean/robust 结果；
- 每轮耗时。

### summary.json

包含最终 PGD-10/20/50 下的汇总结果，是主表数据来源。

---

## 16. 每个指标是什么意思

### Clean macro

先计算每个客户端的 clean accuracy，再对客户端等权平均。

它回答：正常文本输入下，平均每个客户端表现如何？

### Robust macro

先计算每个客户端攻击后的准确率，再对客户端等权平均。

它回答：平均意义上的客户端鲁棒性如何？

### Bottom-20%

选出鲁棒准确率最差的 20% 客户端，再计算它们的平均值。

它是论文最重要的尾部指标，比单独 worst-client 更稳定。

### Worst client

所有客户端中最低的 robust accuracy。

它容易受到单个异常客户端影响，因此必须与 bottom-20% 一起看。

### Client std

客户端 robust accuracy 的标准差。

标准差降低并不一定是好事。如果所有客户端都变差，标准差也可能降低。因此必须先看 bottom-20% 是否提高。

### Conditional ASR

在模型原本 clean 预测正确的样本中，攻击成功使其预测错误的比例。越低越好。

---

## 17. 如何判断方法是否成功

以下情况支持论文主张：

1. FedRDA 的 robust macro 不低于最强 baseline；
2. bottom-20% 和 worst-client 稳定高于 SFAT 和 q-FedAvg-EAT；
3. clean macro 与 FedAvg 和最强鲁棒性 baseline 接近；
4. 三个随机种子的提升方向一致；
5. Qwen2.5-3B 和 7B 的结果方向一致；
6. BERT-Attack 或 DeepWordBug 至少一种具有稳定提升。

以下情况不能支持论文主张：

- 只有 client std 下降；
- worst-client 只在一个 seed 提高；
- clean performance 大幅下降；
- 只对训练时使用的 PGD-3 有效；
- PGD 提高但两个离散攻击都没有改善；
- 结果依赖测试样本很少的客户端；
- 只在 AG News 或只在 3B 上有效，却声称具有通用性。

---

## 18. 自动生成结果表

训练完成后运行：

```bash
$PYTHON_BIN scripts/update_experiment_report.py \
  --outputs outputs \
  --report EXPERIMENTS.md
```

脚本读取每个 `summary.json`，按 run、方法和攻击强度自动合并不同 seed，
报告“均值 ± 样本标准差”。默认排除路径中带 `smoke` 的运行，避免把流水线检查
误当作正式结果。表中的 `n` 应当为 3，`Seeds` 应当是 `42,43,44`；否则说明实验
还没有跑齐。不要手工填写预期结果。

<!-- RESULTS_START -->

| Run group | Method | Attack | n | Seeds | Clean macro | Robust macro | Bottom-20% | Worst client | Client std |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| 尚未运行 | — | — | — | — | — | — | — | — | — |

<!-- RESULTS_END -->

---

## 19. 常见错误

### CUDA out of memory

先检查 GPU 是否被别人使用：

```bash
nvidia-smi
```

如果 GPU 空闲仍然 OOM：

1. 降低 `batch_size`；
2. 降低 `eval_batch_size`；
3. 开启 gradient checkpointing（需要在代码配置中统一记录）；
4. 不要通过减少客户端测试样本逃避 OOM。

### 输出目录已经存在

程序会拒绝覆盖已有的 `round_metrics.jsonl`。

- 如果是同一实验中断：使用 `--resume`；
- 如果是新实验：更换 `--run-name`；
- 不要直接删除一个尚未确认是否有用的正式结果。

### 结果出现 NaN

立即停止该 run，检查：

- learning rate；
- BF16 数值稳定性；
- q-FedAvg denominator；
- 客户端是否只有单一类别；
- 扰动强度是否过大。

不能把 NaN 客户端从结果中悄悄删除。

### SSH 断开

使用 `nohup` 启动的程序一般仍会继续。重新登录后：

```bash
ps -u yhpang -o pid,etime,cmd | grep run_experiment.py
tail -f run_logs/<对应日志文件>
```

### 不知道程序跑到哪一轮

```bash
tail -1 outputs/<run-directory>/round_metrics.jsonl
```

或查看日志：

```bash
grep "round=" run_logs/<log-file> | tail
```

---

## 20. 每完成一个实验必须记录什么

实验完成后，在科研日志中记录：

- 日期；
- Git commit 或当前代码版本；
- GPU 型号；
- config 文件；
- algorithm；
- seed；
- 输出目录；
- 是否发生过中断和续跑；
- 最终 clean/robust/tail/worst；
- 是否运行离散攻击；
- 异常现象和初步解释。

不能只复制一个准确率数字而不记录产生它的配置。

---

## 21. 当前状态

旧实验结果已经删除。新的正式实验尚未运行，因此现在没有可以写入论文的结果，也不能预先声称 FedRDA 优于任何 baseline。
