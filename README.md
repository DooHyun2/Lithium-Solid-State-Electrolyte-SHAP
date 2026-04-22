Lithium Solid State Electrolyte SHAP
SHAP-based interpretability analysis of ionic conductivity in garnet-type solid-state electrolytes using experimentally measured data.

Background
This project extends a previous exploratory study using synthetic data (LLZO-XAI) by applying the same SHAP framework to real experimental data. The goal is to identify which compositional features most strongly influence ionic conductivity in garnet-type solid-state electrolytes.

Dataset

Source: The Liverpool Ionics Dataset

Reference: Hargreaves et al., npj Computational Materials (2022)

Download: Liverpool Ionics Database (requires agreement to terms of use)

Filter applied: Garnet family, room temperature (15–35°C) → 67 samples

Temperature was restricted to minimize thermal effects on ionic conductivity, isolating the influence of composition and dopant type.


Method

Parsed chemical compositions into element-wise stoichiometry features using pymatgen

Trained a Random Forest Regressor on log(ionic conductivity)

Applied SHAP (TreeExplainer) to interpret feature contributions

Model performance

MetricValueTrain R²0.917CV R² (3-fold)0.629 ± 0.185

Results
<img width="1200" height="600" alt="dopant_distribution" src="https://github.com/user-attachments/assets/30a8b57d-6388-450c-b1f6-a4efd8a560e2" />



Significant sample imbalance exists across dopant types (Ta: 32, Nb: 12, Ba: 10, Y: 7, Al: 3).


SHAP Feature Importance (Bar)
<img width="1200" height="1425" alt="shap-importance" src="https://github.com/user-attachments/assets/de50a8e5-060f-4e3e-bb53-8aefcaca569b" />



SHAP Beeswarm Plot

<img width="1200" height="1425" alt="shap_beeswarm" src="https://github.com/user-attachments/assets/8c176d8b-787d-4868-96d5-f7857a8a8814" />



Key Findings

Zr is the dominant factor: lower Zr content (substituted by dopants) tends to decrease ionic conductivity, while higher Zr retention is associated with improved conductivity.

Li stoichiometry is the second most influential feature, consistent with its direct role as the charge carrier.

Ba and Y rank highly among dopants, though their elevated SHAP importance may partly reflect the small number of samples rather than true physical dominance.

Ta, despite being widely regarded as one of the most effective dopants for LLZO, ranks lower — likely due to its high sample count (32) reducing relative variance in the model.


Limitations

Small dataset (67 samples) limits generalizability and causes cross-validation instability.

Dopant sample imbalance (Ta: 32 vs. Al: 3) may bias SHAP rankings.

Compositional features alone do not capture sintering conditions or grain boundary effects.


Future Work

Expand to larger garnet datasets as more experimental data becomes publicly available

Apply Arrhenius correction to include full temperature range

Extend analysis to other electrolyte families (e.g., NASICON, Perovskite)


License

Code: MIT License

Dataset: University of Liverpool — see terms of use
Liverpool Ionics Database

Citation: Hargreaves et al., A database of experimentally measured lithium solid electrolyte conductivities evaluated with machine learning, npj Computational Materials, 2022


