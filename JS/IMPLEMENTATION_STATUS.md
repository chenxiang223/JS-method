# 实施与实验接续记录

更新时间：2026-09-24。本文件是进度记录，不是全套实验完成声明。

## 已完成

- 本地完整模型、显式六域状态接口、七个变体、分类/证据/可靠性训练、独立校准与拒识。
- 原版 datasets 模块从用户 winexp 原工程恢复；原 main.py 导入成功。
- 本地16项测试通过；远端先前15项测试通过，新增的空间整数规划测试在本地通过。
- CPU及远端CUDA合成小样本过拟合通过，eval accuracy=100%；仅为软件链路验证。
- 远端真实PaviaU两epoch全管线通过，包括12组压力测试和状态诊断，不作正式性能结果。
- 已启动正式150epoch实验，每组保留配置、源码哈希、划分、模型、逐样本结果。

## 远端任务

目录 `D:/experiments/jev_sparc_20260924`；环境 `E:/anaconda/envs/LDX/python.exe`（Python3.9，torch2.8+cu128）；两张RTX5070Ti。

数据根目录 `C:/Users/PC/Desktop/cvoca-feinfn/cvoca-feinfn/datasets`。不要更改原数据或原工程。

本地保持SSH的worker会话ID：worker0=72231，worker1=31949。会话可能随工具生命周期变化，以实时PID/心跳/日志为准。不要重复启动正在运行的worker。

```bash
python3 SPARC/scripts/winexp.py status
```

截至最近检查至少21次正式训练完成，后续请使用实时状态。整套任务126次：三数据集×两协议×三种子×七变体。已知固定空间块方案在IndianPines、Salinas上六组划分未通过，预期42个训练任务blocked，84个可运行；最终以manifest核对。

PaviaU空间三个种子已通过标签约束MILP与patch支持不重叠验证，split.npz已提前缓存到远端benchmark/_splits，确保运行中的suite采用它。原先随机搜索全部失败，不能据此声称空间划分不可能；改用约束求解后PaviaU成功。其他两数据集只报告当前固定32×32块方案未获得有效划分，不推论所有空间协议不可能。

## 接续、审计与交付

本任务已有心跳 automationId=`jev-sparc`，每10分钟检查；正常训练期间保持安静，完成/失败/需要操作时通知。完成最终交付后停用心跳。

待两个worker完成后：

```bash
python3 SPARC/scripts/winexp.py fetch --local-output SPARC/outputs/winexp
PYTHONPATH=SPARC .venv/bin/python -m jev.report SPARC/outputs/winexp/outputs/benchmark
PYTHONPATH=SPARC .venv/bin/python SPARC/scripts/audit_results.py SPARC/outputs/winexp/outputs/benchmark
```

fetch会把整个outputs（包括权重）打包取回。模型推理接口在jev/inference.py；完整checkpoint携带标签映射、标准化和校准参数。

最终需完善中文研究总结（结论、三模块贡献/未验证点、分数据集配对差值、校准与拒识表现、压力结果、状态互补性、资源开销、局限），不能以本进度记录代替。

## 必须披露

- 基线是原SPARC模型结构加统一单阶段CE预算，不是原作者完整分阶段长尾训练流程复现。
- 随机协议patch可能重叠。空间协议五个集合的patch支持严格隔离，但部分稀有类无法获得有效划分。
- 独立训练包含CUDA非确定性算子警告，不能保证同seed完全一致。full与no_stress理论上只改变可靠性训练，但独立分类训练已有差异，直接比较存在混杂。最终应明确披露，最好从同一full/best_classifier.pt补充冻结分类器的no_stress对照，不覆盖原结果。
- 校准只拟合干净留出数据，不保证受扰分布的校准；高reliability不是单样本正确性保证。
- 初步IndianPines结果没有显示完整模型稳定优于原SPARC；多状态池化线性头有竞争力。保留负结果，不按测试结果调参。
- 本地snapshot是部分结果快照，不能当作最终完整结果。数值审计重算保存预测，不等同于独立重训复现。

## 工程注意

Windows SSH中的Start-Process脱离会话尝试没有产生训练，故正式worker采用保持SSH连接的前台运行。若网络断开，先检查进程存活和心跳再恢复。

对模型/训练代码修改必须保留每次源码哈希。当前训练启动后仅改动了空间划分fallback及报告脚本；随机训练数值代码未被更改。已有运行的provenance.json记录当时版本。

真实数据没有复制到本地。恢复的datasets原代码与新jev.data分开，正式实验只使用新数据管线，均值/标准差仅拟合训练中心点。
