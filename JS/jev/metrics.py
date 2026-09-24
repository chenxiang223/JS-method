import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score


def ece(scores, correct, bins=15):
    scores,correct = np.asarray(scores),np.asarray(correct)
    ids = np.minimum((scores*bins).astype(int), bins-1)
    table, value = [],0.
    for i in range(bins):
        mask=ids==i
        if mask.any():
            conf,acc = float(scores[mask].mean()),float(correct[mask].mean())
            value += mask.mean()*abs(conf-acc)
            table.append({'bin':i, 'n':int(mask.sum()), 'confidence':conf, 'accuracy':acc})
    return float(value),table


def risk_coverage(scores, errors):
    order=np.argsort(-scores, kind='stable')
    s,e=np.asarray(scores)[order],np.asarray(errors,dtype=float)[order]
    # Expected risk under uniform random ordering within each tied score group.
    starts=np.r_[0,np.flatnonzero(s[1:]!=s[:-1])+1]
    for start,end in zip(starts,np.r_[starts[1:],len(s)]):
        e[start:end]=e[start:end].mean()
    cov=np.arange(1,len(s)+1)/len(s)
    risk=np.cumsum(e)/np.arange(1,len(s)+1)
    return cov,risk


def evaluate(probabilities, labels, reliability, threshold):
    p=np.asarray(probabilities,dtype=np.float64)
    y=np.asarray(labels,dtype=int); r=np.asarray(reliability,dtype=float)
    c=p.shape[1]; pred=p.argmax(1); correct=pred==y; errors=~correct
    cm=confusion_matrix(y,pred,labels=np.arange(c)); support=cm.sum(1)
    per=np.divide(cm.diagonal(),support,out=np.full(c,np.nan),where=support>0)
    oa=float(correct.mean()); expected=float(cm.sum(0)@cm.sum(1)/len(y)**2)
    cal,bins=ece(p.max(1),correct); rcal,rbins=ece(r,correct)
    cov,risk=risk_coverage(r,errors)
    keep=r>=threshold
    idx=np.unique(np.r_[0,np.linspace(0,len(y)-1,min(200,len(y)),dtype=int)])
    result={'n':len(y),'oa':oa,'aa':float(np.nanmean(per)),
            'kappa':float((oa-expected)/(1-expected)) if expected<1 else None,
            'per_class_accuracy':[None if not np.isfinite(v) else float(v) for v in per],
            'support':support.tolist(),'confusion_matrix':cm.tolist(),
            'nll':float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)).mean()),
            'brier':float(((p-np.eye(c)[y])**2).sum(1).mean()),'ece':cal,
            'reliability_ece':rcal,'reliability_brier':float(((r-correct)**2).mean()),
            'error_auroc':float(roc_auc_score(errors,1-r)) if len(np.unique(errors))==2 else None,
            'error_aupr':float(average_precision_score(errors,1-r)) if len(np.unique(errors))==2 else None,
            'aurc':float(risk.mean()),'coverage':float(keep.mean()),
            'selective_risk':float(errors[keep].mean()) if keep.any() else None,
            'threshold':float(threshold), 'probability_bins':bins,'reliability_bins':rbins,
            'risk_coverage':{'coverage':cov[idx].tolist(),'risk':risk[idx].tolist()},
            'undefined_detection_reason':None if len(np.unique(errors))==2 else 'all_predictions_correct_or_all_wrong'}
    return result
