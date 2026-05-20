import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pymatgen.core import Composition
from sklearn.ensemble import RandomForestRegressor
import shap

# Load data
df = pd.read_csv("LiIonDatabase.csv", skiprows=3)
garnet = df[df['family'] == 'Garnet']
garnet_rt = garnet[garnet['temperature'].between(15, 35)].copy().reset_index(drop=True)

def parse_composition(comp_str):
    try:
        comp = Composition(comp_str)
        return {str(el): amt for el, amt in comp.items()}
    except:
        return {}

comp_df = garnet_rt['composition'].apply(parse_composition)
comp_df = pd.DataFrame(comp_df.tolist()).fillna(0)

X = comp_df
y = garnet_rt['log_target'].values

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Zr-Ta dependence plot
shap.dependence_plot("Zr", shap_values, X, interaction_index="Ta", show=False)
plt.tight_layout()
plt.savefig("figures/dependence_Zr_Ta.png", dpi=150)
plt.close()

# Zr-Li dependence plot
shap.dependence_plot("Zr", shap_values, X, interaction_index="Li", show=False)
plt.tight_layout()
plt.savefig("figures/dependence_Zr_Li.png", dpi=150)
plt.close()

print("Saved.")