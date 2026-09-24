"""Recompute reported clean metrics from saved per-sample outputs; no retraining."""
import argparse,json
from pathlib import Path
import numpy as np
from jev.metrics import evaluate
p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();root=Path(a.root)
checked=[]
for path in sorted(root.glob('*/*/seed*/*/result.json')):
    r=json.loads(path.read_text(encoding='utf-8'));z=np.load(path.parent/'predictions.npz')
    again=evaluate(z['probabilities'],z['labels'],z['reliability'],r['calibrated']['threshold'])
    for k in ('oa','aa','kappa','nll','brier','ece','reliability_ece','reliability_brier','error_auroc','error_aupr','aurc','coverage','selective_risk'):
        actual,expected=again[k],r['calibrated'][k]
        if actual is None or expected is None:assert actual is expected,(path,k)
        else:assert np.isclose(actual,expected,rtol=1e-9,atol=1e-10),(path,k,actual,expected)
    assert len(z['centre_ids'])==len(np.unique(z['centre_ids']))
    assert np.allclose(z['probabilities'].sum(1),1,atol=1e-5)
    checked.append(str(path.relative_to(root)))
result={'status':'passed','runs_checked':len(checked),'scope':'clean metrics recomputed from saved predictions; not independent training replication','runs':checked}
(root/'numerical_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(result['status'],len(checked),'runs')
