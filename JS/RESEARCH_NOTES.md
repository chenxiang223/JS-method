# 方法依据与实现边界

本项目是 Jev/System-One 思路启发下的 HSI 决策模型，不调用 Jev API，不声称复现未公开的 Jev 网络。

| 模块 | 本地参考论文 | 实际借鉴与改动 |
|---|---|---|
| 状态压缩 | TokenLearner (NeurIPS 2021), §2, pp.2–3 | 学习输入相关的空间权重；本实现采用 softmax 归一化，每个证据域一个 token，不是原文 sigmoid 后全局均值的逐式复现。 |
| 状态压缩 | SSFTT (TGRS 2022), DOI 10.1109/TGRS.2022.3144158 | 特征到语义 token 的接口；不复刻其 3D/2D CNN 和 Gaussian tokenizer。 |
| 中间表示 | Visual Query Tuning (CVPR 2023) | 保留并聚合中间特征的动机；本模型不宣称实现 VQT 的冻结 ViT 调优。 |
| 输出接口 | Perceiver IO (ICLR 2022) | 异构输入与结构化输出 query 的概念，不堆叠完整 Perceiver 网络。 |
| 类别 query | Query2Label (arXiv:2107.10834v1), §3 | 每类固定绑定一个 query，交叉注意力读取输入相关证据；改为单标签 softmax CE，非原多标签 sigmoid/ASL。 |
| 类别 query | Category Query Learning for HOI Classification (CVPR 2023), §3 | 类别绑定及并行查询思想；不使用 HOI 检测器或 query-as-classifier 交互打分。 |
| 并行输出 | DETR (ECCV 2020) | 一次输出多个 query；本任务没有目标集合匹配，不采用 Hungarian loss。 |
| 分类校准 | Guo et al., On Calibration of Modern Neural Networks (ICML 2017) | 单标量正温度缩放，仅在独立 cal_fit 集最小化 NLL。 |
| 证据学习 | Sensoy et al., Evidential Deep Learning (NeurIPS 2018), §3 | alpha=e+1、Dirichlet 均值平方误差加方差、退火 KL；证据头作为辅助分支，不替换主 softmax 概率。 |
| 失败预测 | Corbière et al., Addressing Failure Prediction by Learning Model Confidence (NeurIPS 2019), §3, pp.4–5 | 分类表示与置信度学习分阶段。原文 TCP 回归并不等于正确率；本项目明确采用正确/错误 BCE，不冒充 ConfidNet 精确复现。 |

## 张量约定

输入 `[B,bands,H,W]` 为仅用训练中心点统计量标准化后的 patch。默认 H=W=15。

`SPARCNet.forward_states(x)` 返回六个 `[B,64,H',W']` 特征图：

- `raw`: `raw_hint_proj(x)`，是原始光谱的学习投影，不是未经变换的反射率。
- `early`: `adapter_out.feature_map`。
- `spatial`: `SpatialHighFrequencyBranch` 的实际输出。
- `frequency`: `DualAxisAmplitudePhaseFrequencyBranch` 的实际输出。
- `fused`: transformer、multi-scale 和 fusion stabilizer 之后的 `mid_feature`。
- `final`: spectral bypass 之后的 `feature`。

原 `forward(x)` 和 `return_aux=True` 的返回契约不变，状态接口不引入参数，原权重可严格加载。缺少空间或频率分支时显式状态接口报错，避免把直通特征冒充独立证据。

状态编译 `[B,6,128]` → 两层四头 query cross-attention + FFN → `[B,C,128]`。不加入新 SPARC backbone block，也不加入深层 Transformer 编码器。

主概率 `softmax(logits/T)`；辅助证据 `softplus(head(D))`；证据不确定性 `C/sum(e+1)`。可靠性读取获胜 token、token 均值、未经温度缩放的最大概率/间隔/熵及证据不确定性，输出一个正确性 logit。校准 `sigmoid(a*logit+b)` 使用正 a，保留排序。

无 evidence 的消融保留零证据及 vacuity=1 作为接口占位，不把它解释为有效的不确定性估计。无可靠性头的对照使用最大 softmax 概率作为选择性预测分数。

## 协议与限制

- 随机协议每类 floor(1%)/floor(5%)/floor(2.5%)/floor(2.5%) 分别为 train/val/cal_fit/cal_threshold；训练至少2点，其余各至少1点，剩余测试。类别少于7点时报错。因此极小类比例会偏离名义比例，split.json 保存实际数量。
- 空间协议：32×32 固定网格，块分区概率为 train=.25、val=.15、cal_fit=.10、cal_threshold=.10、test=.40。patch 必须完整落在相同分区且不越场景边界。每个 seed 先做最多256次固定随机分配，只按类别覆盖计分；若失败，再用仅依赖标签和块内可用中心点的整数规划检查保守可行性（30秒上限），不参考模型输出；每类再按随机协议的名义预算限额采样。记录预算不足、缺失类别和排除中心点，协议之间不声称训练样本完全相同。
- 搜索与约束求解仍未获得所有类别在五个集合的覆盖时，该空间任务标记 blocked_split。这只表示当前固定块方案没有获得有效划分，并非对任意空间划分都不可能的证明。不存在悄悄降低 patch 尺寸或改随机划分的回退。
- 新实验入口直接使用原始可用波段，不用全图自动坏波段检测或全图 PCA，训练中心点均值/标准差是唯一拟合预处理。恢复的原版 datasets 仅用于原入口兼容，新实验使用 jev.data。
- 同一数据/协议/seed 的所有方法共用 split.npz；校准拟合集不用于早停。阈值按 cal_threshold 的90%目标覆盖率选取，边界同分全部保留，实际覆盖率可高于90%。
- 原 SPARC 对照使用原结构、原 cosine/long-tail 头的自动选择，但用统一150 epoch 单阶段 CE 训练；它不是原分阶段长尾训练方法的完整复现。其它变体 CE+0.1 Brier，证据变体另加0.1 EDL。学习率3e-4，AdamW weight_decay=1e-4，cosine schedule，gradient clip=5。评估 batch_size=256，训练 batch_size=64。
- 分类模型只在干净+空间翻转样本训练；可靠性阶段冻结全部分类参数及 BN，用干净和受扰样本训练20 epoch。无压力消融只有干净样本。可靠性标签均为实际当前分类对错。
- 压力测试固定四类扰动各三级，使用所有变体共享的按类别比例抽样测试子集（约4096）；另报告同一子集的干净指标。合成遮挡/噪声只是压力代理，不等于真实传感器漂移或 OOD 保证。
- CKA 与线性探针在固定诊断子集计算，探针只拟合训练表示；类间混淆、互补正确率用于描述，不用于选择测试表现最好的模型。
- 学习正温度和可靠性正尺度限制在 exp([-4,4])；若拟合损失恶化则保留恒等变换。正确性标签只有单一取值时可靠性校准明确退化，不伪造拟合。
- 随机种子是统计重复单位；只报告跨种子的样本标准差，不将海量相关像素当成独立重复。校准不保证跨分布有效。

## 旧数据模块来源

`datasets/*.py` 于本次实现从用户的 winexp 原工程 `C:/Users/PC/Desktop/cvoca-feinfn/cvoca-feinfn/datasets` 恢复。保留原文件内容，仅为补齐本地缺失的原入口依赖；没有下载或复制真实数据到本地。
