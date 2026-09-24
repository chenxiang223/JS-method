# Jev-inspired + SPARC

完整实现：SPARC 六域状态 → Decision State Compiler → Parallel Decision Core → 证据/可靠性 → 校准与拒识。

## 本地安装与验证

在本项目的 `SPARC` 目录运行：

```bash
uv venv ../.venv --python 3.12
uv pip install --python ../.venv/bin/python -r requirements.txt
PYTHONPATH=. ../.venv/bin/python -m pytest tests -q
PYTHONPATH=. ../.venv/bin/python scripts/synthetic_smoke.py
```

测试包括原模型返回/权重兼容、实际分支输出、七种配置梯度与保存恢复、冻结 BN、压力扰动、训练独占归一化、划分隔离、校准退化及指标正确性。合成过拟合只证明链路可学习，不代表真实 HSI 成绩。

## 一个真实实验

```bash
python -m jev.experiment --data-root /path/to/datasets --output outputs/benchmark \
  --dataset PaviaU --protocol random --seed 0 --variant full --device cuda:0
```

数据目录可含原项目的子文件夹。支持 `.npy` 和 `.mat`（包含 v7.3）；明确读取预设键名，路径有歧义时要求缩小 data-root。不会覆盖数据。

## 全部实验

两个终端分别运行，每个数据集/协议/seed 组只属于一个 worker：

```bash
python -m jev.suite --data-root /path/to/datasets --output outputs/benchmark --worker 0 --workers 2 --device cuda:0
python -m jev.suite --data-root /path/to/datasets --output outputs/benchmark --worker 1 --workers 2 --device cuda:1
```

默认三数据集、两协议、三个种子、七个训练变体，共126个计划任务；原 SPARC 温度校准复用同一权重，不重复训练。空间类别覆盖失败会留存 blocked_split，且不启动该组七个训练。每个 worker 遇训练失败即停止并保存错误，不静默重试或终止其它实验。

`status.json` 是每个实验阶段记录；`workerN_status.json` 每10秒记录进程心跳。已完成且配置相同的实验跳过；配置不同或存在未完成结果时拒绝覆盖，需使用新的输出目录保留证据后重跑。

## 输出与推理

每次实验保存配置、源码哈希、环境、固定划分、标准化参数、分类曲线、可靠性曲线、校准参数、最佳分类器、最终模型、测试逐样本预测及12组压力测试结果。

```python
from jev.inference import Predictor
predictor = Predictor('outputs/benchmark/PaviaU/random/seed0/full/model.pt', device='cuda:0')
result = predictor.predict(raw_hwc_patches)  # [B,15,15,bands]，未标准化
# probabilities / original_label / reliability / class_evidence / uncertainty / abstain
```

checkpoint 包含完整分类与校准状态。部署时沿用同一波段顺序、原始数值尺度及 patch 尺寸。`prediction` 是0起始内部类别；`original_label` 还原数据集标签。

汇总结果：

```bash
python -m jev.report outputs/benchmark
```

生成 `results_per_seed.csv`、`summary.json` 和 `REPORT_CN.md`。未运行或阻塞任务不填造结果。方法依据、超参数、对照口径和局限详见 [RESEARCH_NOTES.md](RESEARCH_NOTES.md)。
