import pandas as pd
from pymatgen.core import Composition
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import numpy as np
import shap
import matplotlib.pyplot as plt

# Load dataset and skip header rows
df = pd.read_csv("LiIonDatabase.csv", skiprows=3)

# Filter garnet-type electrolytes
garnet = df[df['family'] == 'Garnet']

# Filter room temperature measurements (15-35°C) to isolate dopant effects
garnet_rt = garnet[garnet['temperature'].between(15, 35)].copy()

print(f"using data: {len(garnet_rt)}개")

# Parse chemical composition into element-wise features
def parse_composition(comp_str):
    try:
        comp = Composition(comp_str)
        return {str(el): amt for el, amt in comp.items()}
    except:
        return {}

comp_df = garnet_rt['composition'].apply(parse_composition)
comp_df = pd.DataFrame(comp_df.tolist()).fillna(0)

# Define features and target (log ionic conductivity)
X = comp_df
y = garnet_rt['log_target'].values

# Train RandomForest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# R² Evaluate model performance with cross-validation
train_r2 = model.score(X, y)
cv_scores = cross_val_score(model, X, y, cv=3, scoring='r2')
print(f"Train R²: {train_r2:.3f}")
print(f"CV R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
print(f"CV folds: {cv_scores.round(3)}")

# Compute SHAP values for feature importance
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Save Beeswarm plot
shap.summary_plot(shap_values, X, show=False)
plt.tight_layout()
plt.savefig("shap_beeswarm.png", dpi=150)
plt.close()

# Save bar plot
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("shap_bar.png", dpi=150)
plt.close()

print("saved")