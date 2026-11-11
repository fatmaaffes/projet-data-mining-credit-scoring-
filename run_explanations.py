"""50-borrower SHAP/LIME concordance with an explicitly named model.

Writes real client-level correlations/overlaps into the specified directory.
Use: python run_explanations.py --output results/explanations
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from src.data import load_german_credit
from src.preprocessing import prepare_data
from src.models import build_model
from run_experiments import explain_agreement


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='results/explanations')
    parser.add_argument('--model',default='Régression logistique')
    parser.add_argument('--n',type=int,default=50)
    args=parser.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    data=load_german_credit()
    prepared=prepare_data(data.X,data.y,random_state=42,test_size=0.20)
    model=build_model(args.model)
    model.fit(prepared.X_train,prepared.y_train)
    result=explain_agreement(model,prepared,args.model,out,n=args.n,seed=42)
    (out/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf8')
    print(json.dumps(result,indent=2,ensure_ascii=False),flush=True)

if __name__=='__main__':main()