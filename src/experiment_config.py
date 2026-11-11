"""Immutable experimental protocol; defaults reproduce the application registry."""
from __future__ import annotations
from dataclasses import asdict,dataclass

MODEL_NAMES = ('Régression logistique','Random Forest','XGBoost','kNN','Arbre de décision')
@dataclass(frozen=True)
class ExperimentConfig:
    seed:int=42
    test_size:float=0.20
    n_splits:int=5
    n_repeats:int=10
    bootstrap_iterations:int=1000
    false_negative_cost:int=5
    false_positive_cost:int=1
    threshold_min:float=.05
    threshold_max:float=.95
    threshold_step:float=.01
    n_explained:int=50
    explanation_model:str='Régression logistique'
    min_group_size:int=10
    
    def as_dict(self):return asdict(self)