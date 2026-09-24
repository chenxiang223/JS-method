"""Aggregate seed-level results, never substitute unrun experiments with estimates."""
import argparse
import csv
import json
from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path
import numpy as np

SCALARS=('oa','aa','kappa','nll','brier','ece','reliability_ece','reliability_brier','error_auroc','error_aupr','aurc','coverage','selective_risk')

def main():
    p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();root=Path(a.root)
    rows=[]; groups=defaultdict(list); runs=[]
    for path in sorted(root.glob('*/*/seed*/*/result.json')):
        data=json.loads(path.read_text(encoding='utf-8')); runs.append(data)
        for stage in ('uncalibrated','calibrated'):
            row={k:data[k] for k in ('dataset','protocol','seed','variant')}
            row['stage']=stage; row.update({k:data[stage][k] for k in SCALARS})
            row.update(parameters=data['compute']['parameters'],batch_ms=data['compute']['batch_ms'],
                       classification_seconds=data['classification_seconds'],reliability_seconds=data['reliability_seconds'])
            rows.append(row);groups[(row['dataset'],row['protocol'],row['variant'],stage)].append(row)
    if rows:
        with (root/'results_per_seed.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=[]
    for key,rs in sorted(groups.items()):
        result=dict(zip(('dataset','protocol','variant','stage'),key));result['seeds']=[r['seed'] for r in rs]
        result['n_seeds']=len(rs)
        for metric in SCALARS:
            values=[r[metric] for r in rs if r[metric] is not None]
            result[metric]={'mean':float(np.mean(values)) if values else None,
                            'std':float(np.std(values,ddof=1)) if len(values)>1 else None,'n':len(values)}
        summary.append(result)
    (root/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    lines=['# Jev + SPARC 实验汇总','', '## Material Passport', '',
           '- Origin Skill: academic-research-suite / experiment-agent',
           '- Origin Mode: run', '- Origin Date: '+datetime.now(timezone.utc).isoformat(),
           '- Verification Status: UNVERIFIED (independent training replication not performed)',
           '- Version Label: jev-sparc-v1', '- Repro Lock: per-run config.json and provenance.json',
           '- Source: user-provided SPARC code and local reference papers; recorded winexp experiments.',
           '- Type: code experiment results; seed-level descriptive analysis.',
           '- Verification: run artifacts and automated tests; no independent repeat-run claim.',
           '- Scope: completed runs only; blocked and pending runs remain explicit.', '', f'已完成独立训练：{len(rows)//2}。统计单位为随机种子；不足三个种子的组为未完成结果。',
           '', '原 SPARC 对照保留原模型结构，使用统一单阶段训练预算；不是原作者完整分阶段训练流程的复现。',
           '', '| 数据集 | 协议 | 模型 | 种子数 | OA | AA | ECE | AURC |', '|---|---|---|---:|---:|---:|---:|---:|']
    def fmt(d):
        if d['mean'] is None:return 'NA'
        return f"{d['mean']:.4f}"+(f" ± {d['std']:.4f}" if d['std'] is not None else '')
    for row in summary:
        if row['stage']!='calibrated':continue
        lines.append('| '+' | '.join([row['dataset'],row['protocol'],row['variant'],str(row['n_seeds']),
                     *[fmt(row[k]) for k in ('oa','aa','ece','aurc')]])+' |')
    blocked=sorted((root/'_splits').glob('*/*/seed*/blocked_split.json'))
    lines+=['','## 空间划分限制','']
    for path in blocked:
        data=json.loads(path.read_text());lines.append(f'- `{path.relative_to(root)}`：缺失类别 {data.get("missing_classes",{})}。未替换为随机划分。')
    if not blocked:lines.append('暂无已记录的划分阻塞。')
    lines += ['', '## 完整模型相对原 SPARC 的配对差值', '',
              '仅比较同数据集、协议和种子的已完成结果。正 OA 差值表示完整模型更高；正 ECE/AURC 差值表示更差。', '',
              '| 数据集 | 协议 | 配对种子数 | ΔOA | ΔECE | ΔAURC |', '|---|---|---:|---:|---:|---:|']
    paired=[]
    for dataset,protocol in sorted({(r['dataset'],r['protocol']) for r in runs}):
        base={r['seed']:r for r in runs if r['dataset']==dataset and r['protocol']==protocol and r['variant']=='sparc'}
        full={r['seed']:r for r in runs if r['dataset']==dataset and r['protocol']==protocol and r['variant']=='full'}
        seeds=sorted(set(base)&set(full))
        if not seeds:continue
        entry={'dataset':dataset,'protocol':protocol,'seeds':seeds}
        cells=[]
        for metric in ('oa','ece','aurc'):
            delta=[full[s]['calibrated'][metric]-base[s]['calibrated'][metric] for s in seeds]
            entry[metric]={'mean':float(np.mean(delta)),'std':float(np.std(delta,ddof=1)) if len(delta)>1 else None,'values':delta}
            cells.append(fmt(entry[metric]))
        paired.append(entry)
        lines.append('| '+' | '.join([dataset,protocol,str(len(seeds)),*cells])+' |')
    (root/'paired_differences.json').write_text(json.dumps(paired,indent=2),encoding='utf-8')
    stress_rows=[]
    for r in runs:
        for corruption,metrics in r['stress'].items():
            stress_rows.append({**{k:r[k] for k in ('dataset','protocol','seed','variant')},'corruption':corruption,
                                **{k:metrics[k] for k in SCALARS}})
    if stress_rows:
        with (root/'stress_per_seed.csv').open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=list(stress_rows[0]));writer.writeheader();writer.writerows(stress_rows)
    lines+=['','## 解读约束','',
        '- 干净测试使用完整测试集；压力测试使用预先固定、按类别比例抽样的最多约 4096 个测试中心点。',
        '- 随机协议允许相邻 patch 重叠，其结果不能作为严格空间泛化证据。',
        '- 校准参数和拒识阈值不使用测试标签；校准集单一正确性标签时保留退化标记。',
        '- 证据不确定性、最大类别概率和正确性概率分别报告；没有证据头的模型不解释其占位不确定性。',
        '- 负结果与正结果同等保留；不根据测试结果反复调参。']
    (root/'REPORT_CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'{len(rows)//2} completed runs; report: {root / "REPORT_CN.md"}')

if __name__=='__main__':main()
