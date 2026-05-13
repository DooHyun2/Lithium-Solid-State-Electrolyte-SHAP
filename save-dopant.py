import pandas as pd
from pymatgen.core import Composition
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error
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

# LOOCV
loo = LeaveOneOut()
y_pred_loo = cross_val_predict(model, X, y, cv=loo)
loo_r2 = r2_score(y, y_pred_loo)
loo_mae = mean_absolute_error(y, y_pred_loo)
print(f"LOOCV R²:       {loo_r2:.3f}")
print(f"LOOCV MAE:      {loo_mae:.3f}  (log10 σ units)")
 
# Parity plot (LOOCV predicted vs actual)
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y, y_pred_loo, alpha=0.6, s=40)
lim = [min(y.min(), y_pred_loo.min()) - 0.5, max(y.max(), y_pred_loo.max()) + 0.5]
ax.plot(lim, lim, 'k--', alpha=0.5, label='y = x')
ax.set_xlabel('Actual log₁₀(σ)')
ax.set_ylabel('LOOCV Predicted log₁₀(σ)')
ax.set_title(f'LOOCV Parity Plot  (R² = {loo_r2:.3f}, N = {len(y)})')
ax.legend()
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('figures/loocv_parity.png', dpi=150, bbox_inches='tight')
plt.close()

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
plt.savefig("shap-importance.png", dpi=150)
plt.close()
# Average SHAP Contribution by Dopant (all dopants, N annotated, low-N faded)

# Sample counts per element
sample_counts = (X > 0).sum(axis=0)

# Exclude LLZO backbone elements
backbone = ['Zr', 'Li', 'La', 'O']
dopants = [el for el in X.columns if el not in backbone]

# Compute signed mean SHAP for all dopants
shap_df = pd.DataFrame(shap_values, columns=X.columns)
mean_shap = shap_df[dopants].mean().sort_values()

# Plot
fig, ax = plt.subplots(figsize=(8, 9))
labels = [f"{d} (N={sample_counts[d]})" for d in mean_shap.index]
bars = ax.barh(labels, mean_shap.values)

# Color by sign, alpha by sample sufficiency
RELIABILITY_THRESHOLD = 5
for bar, d in zip(bars, mean_shap.index):
    color = 'steelblue' if mean_shap[d] > 0 else 'salmon'
    alpha = 1.0 if sample_counts[d] >= RELIABILITY_THRESHOLD else 0.3
    bar.set_color(color)
    bar.set_alpha(alpha)

ax.axvline(0, color='gray', linewidth=0.5)
ax.set_xlabel("Mean SHAP Value")
ax.set_title(f"Average SHAP Contribution by Dopant\n(faded bars: N < {RELIABILITY_THRESHOLD}, insufficient statistical support)")
plt.tight_layout()
plt.savefig("dopant_shap_with_reliability.png", dpi=200)
plt.close()

# # Print reliable vs low-N dopant summary
reliable = [d for d in dopants if sample_counts[d] >= RELIABILITY_THRESHOLD]
low_n = [d for d in dopants if sample_counts[d] < RELIABILITY_THRESHOLD]
print(f"Reliable dopants (N≥{RELIABILITY_THRESHOLD}):", reliable)
print(f"Low-N dopants (excluded from interpretation):", low_n)
print(f"\nReliable mean SHAP (sorted):")
print(shap_df[reliable].mean().sort_values())
print("saved")
