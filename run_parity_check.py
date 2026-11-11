"""Check that Streamlit fixed-protocol estimator and research pipeline are identical.

Calls exactly the src.training path used by the UI and src.evaluation_robust
pipeline used in research; a failing check prevents silent metric divergence.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from src.data import load_german_credit
from src.preprocessing import prepare_data
from src.training import train_one_model
from src.models import build_model
from src.evaluation_robust import make_pipeline
from src.experiment_config import ExperimentConfig,MODEL_NAMES,reference_preparation_config


def run(output):
    data=load_german_credit()
    conf=ExperimentConfig()
    kw=reference_preparation_config(data.X.columns)
    prep=prepare_data(data.X,data.y,test_size=kw['test_size'],random_state=conf.seed,
                      scale_numeric=kw['scale_numeric'],numeric_imputation=kw['numeric_imputation'],
                      categorical_imputation=kw['categorical_imputation'])
    rows=[]
    for name in MODEL_NAMES:
        # Streamlit UI's train_one_model(...tune=False,params={}) code path.
        ui=train_one_model(name,prep,tune=False,params={})
        # Scientific runner's full pipeline fit from *raw* training data.
        batch=make_pipeline(prep.preprocessor,build_model(name))
        batch.fit(prep.X_train_raw,prep.y_train)
        probabilities=batch.predict_proba(prep.X_test_raw)[:,1]
        diff=float(np.max(np.abs(np.asarray(ui.y_prob)-probabilities)))
        rows.append({'model':name,'max_probability_difference':diff,
                     'parity_passed':bool(np.allclose(ui.y_prob,probabilities,rtol=0,atol=1e-9))})
        print(f'{name}: maximum deviation {diff:.3g}',flush=True)
    if not all(row['parity_passed'] for row in rows):
        raise AssertionError('Interactive reference models differ from batch protocol')
    output=Path(output);output.mkdir(exist_ok=True,parents=True)
    (output/'streamlit_batch_parity.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf-8')
    return rows

if __name__=='__main__':
    run('results/parity')