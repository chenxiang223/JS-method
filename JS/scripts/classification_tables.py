"""Paper-style per-class tables from completed results, with paired seed sets."""
import argparse,ast,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--output',required=True);a=p.parse_args()
root=Path(a.root)
tree=ast.parse((Path(__file__).resolve().parents[1]/'main.py').read_text())
for node in tree.body:
    if isinstance(node,(ast.Assign,ast.AnnAssign)):
        targets=node.targets if isinstance(node,ast.Assign) else [node.target]
        if any(isinstance(t,ast.Name) and t.id=='DATASET_CLASS_NAME_PRESETS' for t in targets):names=ast.literal_eval(node.value)
variants=['sparc','pooled_linear','query','full','no_adaptive','no_evidence','no_stress']
labels={'sparc':'SPARC*','pooled_linear':'多状态+线性','query':'状态+Query','full':'完整模型','no_adaptive':'去自适应池化','no_evidence':'去证据监督','no_stress':'去压力训练'}
runs={}
for path in root.glob('*/*/seed*/*/result.json'):
    d=json.loads(path.read_text());runs[d['dataset'],d['protocol'],d['variant'],d['seed']]=d
lines=['# 已完成实验：分类结果对比表','',
       '快照来源：winexp已生成的result.json；仅计入完整训练、校准、测试全部完成的运行。',
       '数值单位均为%，Kappa亦乘100。多种子采用均值±样本标准差（ddof=1，与既定实验计划一致；参考论文采用ddof=0）。单种子只列单次值。',
       'SPARC*表示原SPARC结构在本次统一150 epoch单阶段CE协议下的对照，不是论文原始180+20+20阶段流程的复现。',
       '随机协议名义每类1%训练（至少2个），5%验证，2.5%校准拟合，2.5%阈值选择，其余测试；相邻patch可能重叠。',
       '分类指标覆盖全部测试样本，不是拒识后保留样本的选择性准确率。温度缩放不改变分类argmax。','']
for dataset in ('IndianPines','PaviaU','Salinas'):
    lines+=['## '+dataset,'']
    protocols=[protocol for protocol in ('random','spatial') if any(k[0]==dataset and k[1]==protocol for k in runs)]
    if not protocols:
        lines+=['暂无已完成正式结果，不填入估计值。',''];continue
    for protocol in protocols:
        available={v:sorted(k[3] for k in runs if k[:3]==(dataset,protocol,v)) for v in variants}
        used=[v for v in variants if available[v]]
        seeds=sorted(set.intersection(*(set(available[v]) for v in used)))
        lines+=['### '+('随机像素划分' if protocol=='random' else '严格空间划分'),'']
        lines+=['已完成种子：'+'；'.join(f'{labels[v]}={available[v]}' for v in used)+'。','']
        if not seeds:lines+=['尚无所有已完成方法共有的种子，暂不生成横向比较表。',''];continue
        lines+=['下表统一使用共同种子 '+str(seeds)+'，各列 n='+str(len(seeds))+'。','']
        lines+=['| 类别/指标 | '+' | '.join(labels[v] for v in used)+' |', '|---|'+'---:|'*len(used)]
        def values(v,index=None,metric=None):
            return [100*(runs[dataset,protocol,v,s]['calibrated'][metric] if metric else runs[dataset,protocol,v,s]['calibrated']['per_class_accuracy'][index]) for s in seeds]
        def cell(x):
            return f'{np.mean(x):.2f} ± {np.std(x,ddof=1):.2f}' if len(x)>1 else f'{x[0]:.2f}'
        for i,name in enumerate(names[dataset]):
            lines+=['| '+str(i+1)+' '+name.replace('_','-')+' | '+' | '.join(cell(values(v,index=i)) for v in used)+' |']
        for metric in ('oa','aa','kappa'):
            lines+=['| **'+metric.upper()+'** | '+' | '.join('**'+cell(values(v,metric=metric))+'**' for v in used)+' |']
        lines+=['','未完成方法：'+('、'.join(labels[v] for v in variants if not available[v]) or '无')+'。','']
        # Also publish every available aggregate, without presenting unequal seed sets as paired.
        lines+=['所有已完成运行的汇总（种子数可能不同，不能视作配对比较）：','',
                '| 方法 | 种子 | OA (%) | AA (%) | Kappa (%) |','|---|---|---:|---:|---:|']
        for v in used:
            row=[]
            for metric in ('oa','aa','kappa'):
                row.append(cell([100*runs[dataset,protocol,v,s]['calibrated'][metric] for s in available[v]]))
            lines+=['| '+labels[v]+' | '+str(available[v])+' | '+' | '.join(row)+' |']
        lines+=['']
lines+=['## 阅读边界','',
        '- 这些列是本项目的基线与消融，不是论文中的SF、FDSSC、SSFTT等外部方法复现。未运行的方法没有填入论文成绩。',
        '- full与no_stress采用独立分类训练，CUDA非确定性可能导致分类权重不同，因此分类准确率之差不能归因于压力训练本身；压力训练仅作用于冻结分类器后的可靠性头。',
        '- 尚在运行的结果会变化，本文件只反映本次快照。']
Path(a.output).write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines))
