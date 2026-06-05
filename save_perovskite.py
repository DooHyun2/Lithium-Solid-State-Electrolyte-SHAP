import pandas as pd
from pymatgen.core import Composition
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict, LeaveOneGroupOut

from sklearn.metrics import r2_score, mean_absolute_error

df = pd.read_csv("LiIonDatabase.csv", skiprows=3)

perov = df[df['family'] == 'Perovskite']
perov_rt = perov[perov['temperature'].between(15, 35)].copy()
print(f"Perovskite room temp samples: {len(perov_rt)}")

def parse_composition(comp_str):
    try:
        from pymatgen.core import Composition
        comp = Composition(comp_str)
        return {str(el): amt for el, amt in comp.items()}
    except:
        return {}

comp_df = perov_rt['composition'].apply(parse_composition)
comp_df = pd.DataFrame(comp_df.tolist()).fillna(0)

X = comp_df
y = perov_rt['log_target'].values

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

print(f"Train R²: {model.score(X, y):.3f}")

loo = LeaveOneOut()
y_pred_loo = cross_val_predict(model, X, y, cv=loo)
loo_r2 = r2_score(y, y_pred_loo)
loo_mae = mean_absolute_error(y, y_pred_loo)
print(f"LOOCV R²: {loo_r2:.3f}")
print(f"LOOCV MAE: {loo_mae:.3f}")

# Grouped LOO: hold out all rows sharing a canonical composition
groups = perov_rt['composition'].str.strip().apply(lambda c: Composition(c).reduced_formula).values
y_pred_grouped = cross_val_predict(model, X, y, cv=LeaveOneGroupOut(), groups=groups)
print(f"Grouped LOO R²: {r2_score(y, y_pred_grouped):.3f}")