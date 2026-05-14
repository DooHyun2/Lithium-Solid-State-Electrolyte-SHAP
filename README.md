# Lithium Solid State Electrolyte SHAP

SHAP-based interpretability analysis of ionic conductivity in garnet-type solid-state electrolytes using experimentally measured data.

**Author**: KIM DUHYUN 

**Related**: [LLZO-Synthetic predecessor project](https://github.com/DooHyun2/LLZO-Synthetic)

<img width="600" height="700" alt="shap_beeswarm" src="https://github.com/user-attachments/assets/5e8e8e19-29d0-45b6-aa59-767d5d0e6fa2" />

*Figure 1: SHAP beeswarm plot showing feature contributions to ionic conductivity. 

All 20 element-wise compositional features are displayed for transparency; the signal is concentrated in the top 5 (Zr, Li, Ba, Y, Te), while features below contribute negligibly (|mean SHAP| < 0.05).

Zr's dominant magnitude partly reflects its role as a proxy for B-site substitution — Zr ≈ 0 corresponds to fully substituted garnet variants — rather than a direct chemical effect alone. 

Results are based on N=67 samples interpret as preliminary rankings.*

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

## Model performance

| Metric                    | Value             |
|---------------------------|-------------------|
| Train R²                  | 0.917             |
| LOOCV R²                  | 0.698             |
| LOOCV MAE                 | 0.445 (log10 σ)   |
| 3-fold CV R² (reference)  | 0.629 ± 0.185     |

LOOCV aggregates leave-one-out predictions across all 67 samples

into a single R², providing a more stable estimate than 3-fold CV

on this small dataset. The Train–LOOCV gap (≈0.22) is expected given N=67 samples paired with ~20 element-wise compositional features.

Combined with the 3-fold CV variance (±0.185, attributable to small fold sizes of ~22 samples),

SHAP feature rankings reported below should be interpreted as preliminary patterns warranting cautious treatment,not definitive conclusions. 

In linear conductivity units, MAE = 0.445 corresponds to a factor of ≈2.8× typical prediction error

generalization without indicating severe overfitting.

<img width="500" height="511" alt="loocv_parity" src="https://github.com/user-attachments/assets/b3576712-c77e-498f-ae47-7c8be8fdbd38" />

*Figure: LOOCV parity plot. The model captures the overall trend (R²=0.698, MAE=0.445 in log10 σ units).

A mild regression-to-the-mean tendency is visible in the low-conductivity range (log10 σ ≈ −6 to −5), where predictions are systematically slightly higher than actual values — a known behavior on small datasets. 

One notable outlier near log10 σ ≈ −8 corresponds to an unusually low-conductivity garnet variant.*




## Results

### Dopant Distribution

<img width="600" height="300" alt="dopant_distribution" src="https://github.com/user-attachments/assets/30a8b57d-6388-450c-b1f6-a4efd8a560e2" />



Significant sample imbalance exists across dopant types (Ta: 32, Nb: 12, Ba: 10, Y: 7, Al: 3).



### SHAP Feature Importance (Bar)

<img width="450" height="595" alt="shap-importance" src="https://github.com/user-attachments/assets/de50a8e5-060f-4e3e-bb53-8aefcaca569b" />


### Average SHAP Contribution by Dopant

<img width="560" height="630" alt="dopant_shap_with_reliability" src="https://github.com/user-attachments/assets/5fc31631-927e-432e-b208-b5e7493da53f" />

Figure: Mean SHAP value for each element-wise feature. Zr's dominant

magnitude partly reflects its role as a structural proxy — Zr content

varies primarily through B-site substitution by dopants (Ta, Nb, etc.),

so high Zr SHAP captures the substitution gradient rather than an

isolated chemical effect of Zr itself.


### SHAP Dependence Plots

<img width="600" height="420" alt="dependence_Zr_Li" src="https://github.com/user-attachments/assets/edd0eaf7-43f7-4b11-8c6b-77dadc7f4a34" />


Figure: Zr content vs. SHAP value, colored by Li stoichiometry. Higher Zr retention is consistently associated with positive SHAP values, indicating improved ionic conductivity.


<img width="600" height="420" alt="dependence_Zr_Ta" src="https://github.com/user-attachments/assets/6d9937ca-2794-4114-8933-8171c87aa7df" />


Figure: Zr content vs. SHAP value, colored by Ta content. Samples with Zr≈0 (high Ta substitution) cluster in the negative SHAP region, suggesting excessive Ta substitution may reduce ionic conductivity.

Note: Samples with Zr≈0 represent fully B-site substituted garnets 
where Zr is completely replaced, corresponding to the extreme end 
of the dopant substitution spectrum.

caveat — confounding: Because Zr≈0 samples are necessarily high-Ta (Zr is fully replaced by Ta in those samples), the negative SHAP cluster cannot be unambiguously attributed to Ta excess alone. 

Zr absence and Ta excess are structurally inseparable in this subset, so this plot reflects their combined effect rather than Ta's independent contribution. 

Disentangling the two would require samples with high Ta but partial Zr retention, which are not present in this filtered dataset.

## Key Findings

Zr is the dominant factor: lower Zr content (substituted by dopants) 
tends to decrease ionic conductivity, while higher Zr retention is 
associated with improved conductivity. 

Note: Zr content reduction 
reflects the degree of B-site substitution rather than a direct effect 
of Zr itself.

Li stoichiometry ranks second by SHAP magnitude, consistent with its known role as the charge carrier — though the model captures statistical correlation rather than directly demonstrating the underlying transport mechanism.

Ba and Y rank highly among dopants, though their elevated SHAP 
importance may partly reflect sample imbalance rather than true 
physical dominance.

Ta ranks lower despite being widely regarded as one of the most 
effective LLZO dopants — likely due to high variance across 
Ta-doped samples in this dataset.

Among adequately sampled dopants (N ≥ 5: Ta, Nb, Ba, Y, Sr, Sb), Nb shows the most positive mean SHAP contribution (+0.0069, N=12), while Ba shows the most negative (−0.0155, N=10). Ta, despite being the most abundant dopant (N=32) and widely regarded as the most effective LLZO dopant in the literature, shows a slightly negative average (−0.0040) in this dataset — likely reflecting confounding from variable processing conditions not captured by compositional features alone.

Note: Several literature-relevant dopants (Al, Ga, Te, Mg, Hf, etc.) appear in fewer than 5 samples in this filtered subset and are shown in the figure with faded bars for transparency, but their averages are not interpreted due to insufficient sample support.


## Limitations

Small dataset (67 samples) limits generalizability and causes cross-validation instability.

Dopant sample imbalance (Ta: 32 vs. Al: 3) may bias SHAP rankings.

Compositional features alone do not capture sintering conditions or grain boundary effects.


## Future Work

Expand to larger garnet datasets as more experimental data becomes publicly available

Apply Arrhenius correction to include full temperature range

Extend analysis to other electrolyte families (e.g., NASICON, Perovskite)
  
Incorporate physical descriptors (ionic radius, electronegativity) via matminer
  
Compare with baseline models (Ridge, GradientBoosting)


## License

Code: MIT License

Dataset: University of Liverpool — see terms of use
Liverpool Ionics Database

Citation: Hargreaves et al., A database of experimentally measured lithium solid electrolyte conductivities evaluated with machine learning, npj Computational Materials, 2022


