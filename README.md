# Lithium Solid State Electrolyte SHAP

SHAP-based interpretability analysis of ionic conductivity in garnet-type solid-state electrolytes using experimentally measured data.

<img width="600" height="725" alt="shap_beeswarm" src="https://github.com/user-attachments/assets/5e8e8e19-29d0-45b6-aa59-767d5d0e6fa2" />

*Figure 1: SHAP beeswarm plot showing feature contributions to ionic conductivity. Zr and Li dominate, but results are based on N=67 samples — interpret with caution.*

## Background
This project extends a previous exploratory study using synthetic data (LLZO-XAI) by applying the same SHAP framework to real experimental data. The goal is to identify which compositional features most strongly influence ionic conductivity in garnet-type solid-state electrolytes.

## Dataset

Source: The Liverpool Ionics Dataset

Reference: Hargreaves et al., npj Computational Materials (2022)

Download: Liverpool Ionics Database (requires agreement to terms of use)

Filter applied: Garnet family, room temperature (15–35°C) → 67 samples

Temperature was restricted to minimize thermal effects on ionic conductivity, isolating the influence of composition and dopant type.

## Installation

pip install -r requirements.txt


## Method

Parsed chemical compositions into element-wise stoichiometry features using pymatgen

Trained a Random Forest Regressor on log(ionic conductivity)

Applied SHAP (TreeExplainer) to interpret feature contributions

Model performance

MetricValueTrain R²0.917CV R² (3-fold)0.629 ± 0.185

## Results
<img width="1200" height="600" alt="dopant_distribution" src="https://github.com/user-attachments/assets/30a8b57d-6388-450c-b1f6-a4efd8a560e2" />



Significant sample imbalance exists across dopant types (Ta: 32, Nb: 12, Ba: 10, Y: 7, Al: 3).


SHAP Feature Importance (Bar)
<img width="1200" height="1425" alt="shap-importance" src="https://github.com/user-attachments/assets/de50a8e5-060f-4e3e-bb53-8aefcaca569b" />


Average SHAP Contribution by Dopant
<img width="1200" height="750" alt="Contribution" src="https://github.com/user-attachments/assets/9ae311a3-62c5-44aa-8f1e-ec483600e3fa" />


<img width="1125" height="750" alt="dependence_Zr_Li" src="https://github.com/user-attachments/assets/edd0eaf7-43f7-4b11-8c6b-77dadc7f4a34" />
*Figure: Zr content vs. SHAP value, colored by Li stoichiometry. Higher Zr retention is consistently associated with positive SHAP values, indicating improved ionic conductivity.*


<img width="1125" height="750" alt="dependence_Zr_Ta" src="https://github.com/user-attachments/assets/6d9937ca-2794-4114-8933-8171c87aa7df" />
*Figure: Zr content vs. SHAP value, colored by Ta content. Samples with Zr≈0 (high Ta substitution) cluster in the negative SHAP region, suggesting excessive Ta substitution may reduce ionic conductivity.*




## Key Findings

Zr is the dominant factor: lower Zr content (substituted by dopants) 
tends to decrease ionic conductivity, while higher Zr retention is 
associated with improved conductivity. Note: Zr content reduction 
reflects the degree of B-site substitution rather than a direct effect 
of Zr itself.

Li stoichiometry is the second most influential feature, consistent 
with its direct role as the charge carrier.

Ba and Y rank highly among dopants, though their elevated SHAP 
importance may partly reflect sample imbalance rather than true 
physical dominance.

Ta ranks lower despite being widely regarded as one of the most 
effective LLZO dopants — likely due to high variance across 
Ta-doped samples in this dataset.

When isolating dopant-only SHAP contributions, Te, Mg, and Nb show 
positive average values, while Ba and Ta show negative, suggesting 
high variance across dopant concentrations in this dataset.


## Limitations

Small dataset (67 samples) limits generalizability and causes cross-validation instability.

Dopant sample imbalance (Ta: 32 vs. Al: 3) may bias SHAP rankings.

Compositional features alone do not capture sintering conditions or grain boundary effects.


## Future Work

Expand to larger garnet datasets as more experimental data becomes publicly available

Apply Arrhenius correction to include full temperature range

Extend analysis to other electrolyte families (e.g., NASICON, Perovskite)

Add LOOCV for more stable cross-validation on small datasets
  
Incorporate physical descriptors (ionic radius, electronegativity) via matminer
  
Compare with baseline models (Ridge, GradientBoosting)


## License

Code: MIT License

Dataset: University of Liverpool — see terms of use
Liverpool Ionics Database

Citation: Hargreaves et al., A database of experimentally measured lithium solid electrolyte conductivities evaluated with machine learning, npj Computational Materials, 2022


