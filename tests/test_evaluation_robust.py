import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from src.evaluation_robust import bootstrap_auc,total_cost,choose_cost_threshold,subgroup_audit,repeated_cv,make_pipeline


def test_bootstrap_deterministic_and_finite():
    y=np.array([0]*20+[1]*20);p=np.array([.15]*20+[.85]*20)
    a=bootstrap_auc(y,p,n=100,seed=7);b=bootstrap_auc(y,p,n=100,seed=7)
    assert a==b
    assert a['ci_low']==a['ci_high']==1


def test_cost_asymmetric():
    y=[0,0,1,1];pred=[1,0,0,1]
    assert total_cost(y,pred)==6
    threshold,table=choose_cost_threshold(y,[.2,.2,.4,.9],thresholds=[.3,.5],c_fn=5,c_fp=1)
    assert threshold==.3


def test_subgroup_denominators():
    y=[0,0,1,1];p=[.1,.9,.2,.8];g=['A','A','A','A']
    d=subgroup_audit(y,p,g,threshold=.5,min_support=1)
    assert d.iloc[0].FPR==.5 and d.iloc[0].FNR==.5


def test_repeated_cv_pipeline():
    rng=np.random.default_rng(42);X=pd.DataFrame({'amount':rng.normal(size=60),
      'cat':rng.choice(['x','y'],size=60)});y=np.array([0,1]*30)
    preprocess=ColumnTransformer([('numeric',StandardScaler(),['amount']),
          ('categorical',OneHotEncoder(handle_unknown='ignore'),['cat'])])
    pipe=make_pipeline(preprocess,LogisticRegression(max_iter=1000))
    scores=repeated_cv(X,y,pipe,n_splits=3,n_repeats=2,seed=42)
    assert len(scores)==6
    assert set(scores)=={'ROC-AUC','Average Precision','F1@0.50'}