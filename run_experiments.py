"""Reproducible, leakage-aware experiments for the Statlog German Credit study.

python run_experiments.py --output results/research
python run_experiments.py --output results/research --with-explanations
python run_experiments.py --data-path /path/to/german.data --output results/research

The TEST is used once for final prediction/evaluation; all tuning/thresholds are
fit on TRAIN. Repeated CV evaluates stability on TRAIN (not data-derived model selection).
No model is tuned to the final TEST. Subgroup results are descriptive only.
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import json
from pathlib import Path
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (roc_curve,precision_recall_curve,roc_auc_score,
                             average_precision_score,PrecisionRecallDisplay)
from sklearn.calibration import calibration_curve
from src.data import COLUMN_NAMES,load_german_credit
from src.preprocessing import prepare_data
from src.models import build_model
from src.experiment_config import ExperimentConfig, MODEL_NAMES
from src.evaluation_robust import (bootstrap_auc,cost_breakdown,choose_cost_threshold,
    make_pipeline,repeated_cv,training_oof_probabilities,subgroup_audit,brier_score)


def load_data(path):
    if not path:
        d=load_german_credit()
        return d.X,d.y
    columns=list(COLUMN_NAMES.values())+['original_class']
    raw=pd.read_csv(path,sep=r'\s+',header=None,names=columns)
    y=raw.pop('original_class').map({1:0,2:1}).rename('default')
    if y.isna().any() or raw.shape!=(1000,20):
        raise ValueError('Not the UCI Statlog German Credit german.data original dataset')
    return raw,y


def save_plot(fig,directory,name):
    fig.savefig(directory/(name+'.png'),dpi=190,bbox_inches='tight')
    fig.savefig(directory/(name+'.svg'),bbox_inches='tight')
    plt.close(fig)


def build_roc_plot(y_test,probabilities,directory):
    fig,ax=plt.subplots(figsize=(7.5,5.4))
    for name,p in probabilities.items():
        fpr,tpr,_=roc_curve(y_test,p)
        ax.plot(fpr,tpr,label=f'{name} (AUC {roc_auc_score(y_test,p):.3f})')
    ax.plot([0,1],[0,1],'--',color='gray',label='Random ranking')
    ax.set(xlabel='False positive rate',ylabel='True positive rate',title='ROC curves — held-out test')
    ax.legend(fontsize=8);fig.tight_layout();save_plot(fig,directory,'roc_curves')


def build_pr_plot(y_test,probabilities,directory):
    fig,ax=plt.subplots(figsize=(7.5,5.4))
    for name,p in probabilities.items():
        precision,recall,_=precision_recall_curve(y_test,p)
        ax.step(recall,precision,where='post',label=f'{name} (AP {average_precision_score(y_test,p):.3f})')
    ax.axhline(float(np.asarray(y_test).mean()),linestyle='--',color='gray',label='Positive prevalence')
    ax.set(xlabel='Recall',ylabel='Precision',title='Precision–recall curves — held-out test',xlim=(0,1),ylim=(0,1.03))
    ax.legend(fontsize=8);fig.tight_layout();save_plot(fig,directory,'precision_recall_curves')


def build_cost_plot(cost_frame,directory):
    fig,ax=plt.subplots(figsize=(9,4.7))
    names=cost_frame.model.tolist(); x=np.arange(len(names));w=.36
    ax.bar(x-w/2,cost_frame.cost_050,width=w,label='Threshold 0.50')
    ax.bar(x+w/2,cost_frame.cost_optimised,width=w,label='Train-OOF cost threshold')
    ax.set_xticks(x,names,rotation=20,ha='right');ax.set_ylabel('Test cost (FN×5 + FP×1)')
    ax.set_title('Out-of-sample classification cost');ax.legend();fig.tight_layout()
    save_plot(fig,directory,'test_cost_comparison')


def build_confusion(y,proba,threshold,directory,label):
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    matrix=confusion_matrix(y,(np.asarray(proba)>=threshold).astype(int),labels=[0,1])
    fig,ax=plt.subplots(figsize=(4.4,3.9)); im=ax.imshow(matrix,cmap='Blues')
    ax.set_xticks([0,1],['Good','Default']);ax.set_yticks([0,1],['Good','Default'])
    ax.set(xlabel='Predicted',ylabel='Actual',title=f'Confusion matrix — {label} (τ={threshold:.2f})')
    for (i,j),v in np.ndenumerate(matrix):ax.text(j,i,str(v),ha='center',va='center',color='white' if v>matrix.max()/2 else 'black')
    fig.colorbar(im,ax=ax,shrink=.75);fig.tight_layout();save_plot(fig,directory,'confusion_logistic_cost')


def build_calibration(y,p,directory):
    prob_true, prob_pred=calibration_curve(y,p,n_bins=8,strategy='quantile')
    fig,ax=plt.subplots(figsize=(5.5,4.5))
    ax.plot([0,1],[0,1],'--',color='gray',label='Perfect calibration')
    ax.plot(prob_pred,prob_true,'o-',label='Logistic regression')
    ax.set(xlabel='Mean predicted default probability',ylabel='Observed default frequency',
           xlim=(0,1),ylim=(0,1),title='Calibration — held-out test')
    ax.legend();fig.tight_layout();save_plot(fig,directory,'calibration_logistic')


def explain_agreement(model,prepared,model_name,folder,seed=42,n=50):
    """Compare model-agnostic permutation SHAP vs LIME on aligned OHE columns.

    The LIME discretisation and its local linear surrogate mean contribution
    scales are NOT commensurable. Spearman and top-k overlap are descriptive.
    Omitted LIME features have zero contribution in the recorded sparse vector.
    """
    import shap
    from scipy.stats import spearmanr
    from lime.lime_tabular import LimeTabularExplainer
    rng=np.random.default_rng(seed)
    Xtr=prepared.X_train
    Xte=prepared.X_test
    feature_names=list(Xtr.columns)
    sampled=rng.choice(len(Xte),size=min(n,len(Xte)),replace=False)
    bg=Xtr.sample(n=min(30,len(Xtr)),random_state=seed).to_numpy(dtype=float)
    def pred(v):return model.predict_proba(np.asarray(v,dtype=float))[:,1]
    shap_explainer=shap.Explainer(pred,bg,feature_names=feature_names,algorithm='permutation',seed=seed)
    lime=LimeTabularExplainer(Xtr.to_numpy(dtype=float),feature_names=feature_names,
        class_names=['Good','Default'],mode='classification',discretize_continuous=True,
        random_state=seed)
    rows=[]
    for index in sampled:
        row=Xte.iloc[index].to_numpy(dtype=float)
        s=np.asarray(shap_explainer(row.reshape(1,-1),max_evals=2*len(row)+1).values).reshape(-1)
        explanation=lime.explain_instance(row,pred,num_features=len(row),num_samples=1000)
        local=np.zeros(len(row),dtype=float)
        for i,value in explanation.as_map()[1]:local[i]=value
        rank_corr=spearmanr(s,local).statistic
        top_s=set(np.argsort(np.abs(s))[-5:]);top_l=set(np.argsort(np.abs(local))[-5:])
        rows.append({'test_row':int(Xte.index[index]),'model':model_name,
                     'spearman':float(rank_corr) if np.isfinite(rank_corr) else np.nan,
                     'top5_overlap':len(top_s & top_l)/5,
                     'default_probability':float(pred(row.reshape(1,-1))[0])})
    result=pd.DataFrame(rows);result.to_csv(folder/'shap_lime_client_agreement.csv',index=False)
    return {'model':model_name,'n':len(result),'spearman_mean':float(result.spearman.mean()),
            'spearman_std':float(result.spearman.std(ddof=1)),
            'top5_overlap_mean':float(result.top5_overlap.mean()),
            'top5_overlap_std':float(result.top5_overlap.std(ddof=1))}


def run(args):
    config=ExperimentConfig(n_splits=args.folds,n_repeats=args.repeats,
          bootstrap_iterations=args.bootstrap,n_explained=args.explain_cases)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    plots=output/'figures';plots.mkdir(exist_ok=True)
    X,y=load_data(args.data_path)
    prepared=prepare_data(X,y,test_size=config.test_size,random_state=config.seed,
                          scale_numeric=True)
    config_dump={'protocol':config.as_dict(),'models':{},
      'source':'UCI Statlog German Credit original categorical dataset',
      'note':'Cost=5 FN + 1 FP, theoretical threshold 1/6 ONLY if probability is calibrated',
      'dataset_n':len(X),'train_n':len(prepared.y_train),'test_n':len(prepared.y_test),
      'python_versions':{p:md.version(p) for p in ['pandas','numpy','scikit-learn','scipy','matplotlib','xgboost']}}
    cv_rows=[];ci_rows=[];cost_rows=[];preds={};fitted={}
    for name in MODEL_NAMES:
        print('Running:',name,flush=True)
        estimator=build_model(name)
        config_dump['models'][name]=estimator.get_params()
        pipe=make_pipeline(prepared.preprocessor,estimator)
        cv_df=repeated_cv(prepared.X_train_raw,prepared.y_train,pipe,
              n_splits=config.n_splits,n_repeats=config.n_repeats,seed=config.seed)
        for metric in cv_df:
            cv_rows.append({'model':name,'metric':metric,'mean':float(cv_df[metric].mean()),
                            'std':float(cv_df[metric].std(ddof=1)),
                            'n_folds':len(cv_df),'sample':'TRAIN only'})
        oof=training_oof_probabilities(prepared.X_train_raw,prepared.y_train,pipe,
                             folds=config.n_splits,seed=config.seed)
        chosen,threshold_grid=choose_cost_threshold(prepared.y_train,oof,
                        c_fn=config.false_negative_cost,c_fp=config.false_positive_cost)
        threshold_grid.to_csv(output/f'train_cost_grid_{MODEL_NAMES.index(name)}.csv',index=False)
        pipe.fit(prepared.X_train_raw,prepared.y_train)
        p=pipe.predict_proba(prepared.X_test_raw)[:,1]
        preds[name]=p;fitted[name]=pipe
        ci=bootstrap_auc(prepared.y_test,p,n=config.bootstrap_iterations,seed=config.seed)
        ci_rows.append({'model':name,**ci, 'average_precision':float(average_precision_score(prepared.y_test,p)),
                        'brier':brier_score(prepared.y_test,p)})
        baseline=cost_breakdown(prepared.y_test,p,.50,5,1)
        optimized=cost_breakdown(prepared.y_test,p,chosen,5,1)
        cost_rows.append({'model':name,'selected_threshold_train':chosen,
              'cost_050':baseline['cost'],'cost_optimised':optimized['cost'],
              'FN_050':baseline['FN'],'FP_050':baseline['FP'],
              'FN_optimised':optimized['FN'],'FP_optimised':optimized['FP'],
              'recall_050':baseline['recall'],'recall_optimised':optimized['recall']})
        print(name,'test AUC',round(ci['auc'],4),'CI',round(ci['ci_low'],4),round(ci['ci_high'],4),
              'test costs',baseline['cost'],optimized['cost'],flush=True)
    cv=pd.DataFrame(cv_rows);cv.to_csv(output/'repeated_cv_train.csv',index=False)
    pd.DataFrame(ci_rows).to_csv(output/'bootstrap_test_auc.csv',index=False)
    costs=pd.DataFrame(cost_rows);costs.to_csv(output/'test_costs.csv',index=False)
    pd.DataFrame({'actual_default':prepared.y_test.to_numpy(),**preds},index=prepared.X_test_raw.index).to_csv(output/'test_predictions.csv',index_label='original_row')
    build_roc_plot(prepared.y_test,preds,plots)
    build_pr_plot(prepared.y_test,preds,plots)
    build_cost_plot(costs,plots)
    best='Régression logistique'
    thr=float(costs.loc[costs.model==best,'selected_threshold_train'].iloc[0])
    build_confusion(prepared.y_test,preds[best],thr,plots,'Logistic regression')
    build_calibration(prepared.y_test,preds[best],plots)
    age=prepared.X_test_raw['age']
    age_group=np.where(age <25,'Age <25','Age ≥25')
    fairness=[]
    for group_col, group in [('personal_status_sex',prepared.X_test_raw['personal_status_sex']),('age_band',age_group)]:
        audit=subgroup_audit(prepared.y_test,preds[best],group,thr,min_support=config.min_group_size)
        audit.insert(0,'grouping',group_col)
        fairness.append(audit)
    pd.concat(fairness,ignore_index=True).to_csv(output/'subgroup_audit.csv',index=False)
    ablation=[]
    for variant,dropped in [('full',[]),('without_personal_status_sex',['personal_status_sex']),
                             ('without_personal_status_sex_and_age',['personal_status_sex','age'])]:
        features=X.drop(columns=dropped)
        p_data=prepare_data(features,y,test_size=config.test_size,random_state=config.seed)
        mdl=build_model(best)
        pipe=make_pipeline(p_data.preprocessor,mdl)
        oof=training_oof_probabilities(p_data.X_train_raw,p_data.y_train,pipe,folds=5,seed=42)
        chosen,_=choose_cost_threshold(p_data.y_train,oof,c_fn=5,c_fp=1)
        pipe.fit(p_data.X_train_raw,p_data.y_train)
        p=pipe.predict_proba(p_data.X_test_raw)[:,1]
        ablation.append({'variant':variant,'excluded_features':', '.join(dropped) or 'none',
             'test_auc':float(roc_auc_score(p_data.y_test,p)),
             'train_selected_threshold':chosen,
             **{('test_'+key):value for key,value in cost_breakdown(p_data.y_test,p,chosen,5,1).items()}})
    pd.DataFrame(ablation).to_csv(output/'ablation_fairness.csv',index=False)
    if args.with_explanations:
        # Distinct exhaustive operation; no more than configured sample, no random test peeking.
        model=build_model(config.explanation_model)
        model.fit(prepared.X_train,prepared.y_train)
        agreement=explain_agreement(model,prepared,config.explanation_model,output,n=config.n_explained)
        (output/'shap_lime_summary.json').write_text(json.dumps(agreement,indent=2),encoding='utf-8')
    (output/'experiment_config.json').write_text(json.dumps(config_dump,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
    print('All outputs:',output.resolve(),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='results/research')
    p.add_argument('--data-path',default=None,help='Optional local UCI german.data file; otherwise download via ucimlrepo')
    p.add_argument('--folds',type=int,default=5)
    p.add_argument('--repeats',type=int,default=10)
    p.add_argument('--bootstrap',type=int,default=1000)
    p.add_argument('--with-explanations',action='store_true',help='Run costly SHAP/LIME agreement on fixed test sample')
    p.add_argument('--explain-cases',type=int,default=50)
    run(p.parse_args())