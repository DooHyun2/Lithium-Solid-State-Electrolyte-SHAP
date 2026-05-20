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

# Extract only dopants (Exception Li, La, Zr, O)
base_elements = ['Li', 'La', 'Zr', 'O']
dopants = [col for col in X.columns if col not in base_elements]

shap_df = pd.DataFrame(shap_values, columns=X.columns)
dopant_shap = shap_df[dopants].mean()
dopant_shap = dopant_shap.sort_values()

plt.figure(figsize=(8, 5))
dopant_shap.plot(kind='barh', color='steelblue')
plt.xlabel('Mean SHAP Value')
plt.title('Average SHAP Contribution by Dopant')
plt.tight_layout()
plt.savefig("figures/Contribution.png", dpi=150)
plt.close()

print("Saved.")