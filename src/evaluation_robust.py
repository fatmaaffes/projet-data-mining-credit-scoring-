"""Uncertainty, cost-sensitive selection and subgroup diagnostics for German Credit.

All threshold choices must be computed from training out-of-fold probabilities.
The test split must be reserved for one final performance assessment.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.base import clone


def bootstrap_auc(y_true, y_proba, n=1000, seed=42):
    """Paired, stratified-data-resampled percentile CI on a held-out test set.

    Resamples cases, not CV folds; does not measure training-set randomness.
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_proba, dtype=float)
    if len(y) != len(p) or len(y) == 0 or len(np.unique(y)) < 2:
        raise ValueError('Expected equal-length arrays containing both classes')
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if np.unique(y[idx]).size == 2:
            values.append(roc_auc_score(y[idx], p[idx]))
    if not values:
        raise ValueError('No valid bootstrap resamples')
    low, high = np.percentile(values, [2.5, 97.5])
    return {'auc': float(roc_auc_score(y,p)), 'bootstrap_mean': float(np.mean(values)),
            'ci_low': float(low), 'ci_high': float(high), 'n_valid': len(values)}


def total_cost(y_true, y_pred, c_fn=5, c_fp=1):
    y = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    if y.shape != pred.shape: raise ValueError('Shape mismatch')
    fn = int(((y == 1) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    return c_fn * fn + c_fp * fp


def cost_breakdown(y_true, y_prob, threshold, c_fn=5, c_fp=1):
    from sklearn.metrics import precision_score, recall_score, f1_score
    pred = (np.asarray(y_prob) >= threshold).astype(int)
    y = np.asarray(y_true, dtype=int)
    return {'threshold': float(threshold), 'cost': float(total_cost(y,pred,c_fn,c_fp)),
            'FN': int(((y == 1) & (pred == 0)).sum()), 'FP': int(((y == 0) & (pred == 1)).sum()),
            'precision': float(precision_score(y,pred,zero_division=0)),
            'recall': float(recall_score(y,pred,zero_division=0)),
            'f1': float(f1_score(y,pred,zero_division=0))}


def choose_cost_threshold(y_true, oof_proba, thresholds=None, c_fn=5, c_fp=1):
    """Select by *training OOF* cost. Deterministic tie-break: largest threshold.

    The theoretical t=c_fp/(c_fp+c_fn) only applies to calibrated probabilities.
    """
    ts = np.linspace(0.05,0.95,91) if thresholds is None else np.asarray(thresholds)
    if len(ts) == 0 or np.any((ts <= 0)|(ts >= 1)): raise ValueError('Invalid thresholds')
    rows = [cost_breakdown(y_true,oof_proba,t,c_fn,c_fp) for t in ts]
    table = pd.DataFrame(rows)
    winner = table.sort_values(['cost','threshold'],ascending=[True,False]).iloc[0]
    return float(winner['threshold']),table


def make_pipeline(preprocessor, classifier):
    return Pipeline([('preprocessor',clone(preprocessor)),('model',clone(classifier))])


def repeated_cv(X, y, pipeline, n_splits=5, n_repeats=10, seed=42, n_jobs=1):
    """Full preprocessing refit inside each stratified CV fold.

    Primary research comparison should use TRAIN ONLY; optional 1000-row CV
    is supplemental and must not be used to select the reported test model.
    """
    cv = RepeatedStratifiedKFold(n_splits=n_splits,n_repeats=n_repeats,random_state=seed)
    values = cross_validate(clone(pipeline),X,y,cv=cv,
        scoring={'ROC-AUC':'roc_auc','Average Precision':'average_precision','F1@0.50':'f1'},
        n_jobs=n_jobs,error_score='raise')
    return pd.DataFrame({label: values['test_'+label] for label in
                         ('ROC-AUC','Average Precision','F1@0.50')})


def training_oof_probabilities(X_train,y_train,pipeline,folds=5,seed=42):
    cv = StratifiedKFold(n_splits=folds,shuffle=True,random_state=seed)
    return cross_val_predict(clone(pipeline),X_train,y_train,cv=cv,
                             method='predict_proba',n_jobs=1)[:,1]


def subgroup_audit(y_true, y_prob, groups, threshold, min_support=10):
    """Descriptive subgroup error rates; small samples are suppressed.

    Group comparisons do not demonstrate fairness or legal compliance.
    """
    y = np.asarray(y_true,dtype=int); p = np.asarray(y_prob,dtype=float)
    g = np.asarray(groups); pred = (p >= threshold).astype(int)
    if not (len(y)==len(p)==len(g)): raise ValueError('Shape mismatch')
    rows=[]
    for group in sorted(set(g.astype(str))):
        mask = g.astype(str)==group
        size = int(mask.sum())
        positive = int(((y==1)&mask).sum()); negative=size-positive
        if size < min_support:
            rows.append({'group':group,'n':size,'n_actual_default':positive,
                         'n_actual_good':negative,'predicted_default_rate':np.nan,
                         'FPR':np.nan,'FNR':np.nan,'note':'suppressed: small subgroup'})
            continue
        fp = int(((y==0)&(pred==1)&mask).sum()); fn=int(((y==1)&(pred==0)&mask).sum())
        rows.append({'group':group,'n':size,'n_actual_default':positive,
                     'n_actual_good':negative,'predicted_default_rate':float(pred[mask].mean()),
                     'FPR':float(fp/negative) if negative else np.nan,
                     'FNR':float(fn/positive) if positive else np.nan,
                     'note':'not estimable' if not (negative and positive) else 'descriptive only'})
    return pd.DataFrame(rows)


def brier_score(y_true,y_prob):
    return float(brier_score_loss(y_true,y_prob))