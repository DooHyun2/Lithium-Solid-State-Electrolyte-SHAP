import pandas as pd
import numpy as np
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error

# Data loading
df = pd.read_csv("LiIonDatabase.csv", skiprows=3)
garnet = df[df['family'] == 'Garnet']
garnet_rt = garnet[garnet['temperature'].between(15, 35)].copy().reset_index(drop=True)

# Original stoichiometry features
def parse_composition(comp_str):
    try:
        comp = Composition(comp_str)
        return {str(el): amt for el, amt in comp.items()}
    except:
        return {}

comp_df = garnet_rt['composition'].apply(parse_composition)
comp_df = pd.DataFrame(comp_df.tolist()).fillna(0).reset_index(drop=True)

# Parse compositions into pymatgen objects
garnet_rt['comp_obj'] = garnet_rt['composition'].apply(lambda x: Composition(x))

# ElementProperty featurizer (Magpie preset)
ep = ElementProperty.from_preset("magpie")
matminer_features = ep.featurize_many(garnet_rt['comp_obj'].tolist(), ignore_errors=True)
matminer_df = pd.DataFrame(matminer_features, columns=ep.feature_labels()).fillna(0)

# Combine stoichiometry and matminer features
X_combined = pd.concat([comp_df, matminer_df], axis=1)
y = garnet_rt['log_target'].values

print(f"전체 feature 수: {X_combined.shape[1]}")

# LOOCV
loo = LeaveOneOut()
model = RandomForestRegressor(n_estimators=100, random_state=42)
y_pred = cross_val_predict(model, X_combined, y, cv=loo)

loocv_r2 = r2_score(y, y_pred)
loocv_mae = mean_absolute_error(y, y_pred)

print(f"기존 LOOCV R²: 0.698")
print(f"matminer 추가 후 LOOCV R²: {loocv_r2:.3f}")
print(f"LOOCV MAE: {loocv_mae:.3f}")

# 핵심 descriptor만 선택 (이온 반지름, 전기음성도, 산화수)
key_features = [col for col in matminer_df.columns if any(
    keyword in col for keyword in [
        'Electronegativity',
        'OxidationState', 
        'AtomicRadius'
    ]
)]

print(f"선택된 feature 수: {len(key_features)}")

X_selected = pd.concat([comp_df, matminer_df[key_features]], axis=1)
y_pred2 = cross_val_predict(model, X_selected, y, cv=loo)
loocv_r2_selected = r2_score(y, y_pred2)
print(f"핵심 descriptor만 추가 후 LOOCV R²: {loocv_r2_selected:.3f}")