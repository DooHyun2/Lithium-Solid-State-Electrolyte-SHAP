# Lithium Solid State Electrolyte SHAP

SHAP-based interpretability analysis of ionic conductivity in lithium solid-state electrolytes across multiple structural 
families (Garnet, NASICON, Perovskite) using experimentally measured data from the Hargreaves database

**Author**: KIM DUHYUN 
**Related**: [LLZO-Synthetic](https://github.com/DooHyun2/LLZO-Synthetic) — SHAP + Bayesian Optimization on synthetic LLZO testbed

<img width="600" height="700" alt="shap_beeswarm" src="https://github.com/user-attachments/assets/5e8e8e19-29d0-45b6-aa59-767d5d0e6fa2" />

Figure: SHAP beeswarm plot showing feature contributions to ionic conductivity. 

All 20 element-wise compositional features are displayed for transparency; the signal is concentrated in the top 5 (Zr, Li, Ba, Y, Te), while features below contribute negligibly (|mean SHAP| < 0.05).

Zr's dominant magnitude partly reflects its role as a proxy for B-site substitution — Zr ≈ 0 corresponds to fully substituted garnet variants — rather than a direct chemical effect alone. 

Results are based on N=67 samples - interpret as preliminary rankings.*

## Background

The goal is to identify which compositional features most strongly 
influence ionic conductivity in lithium solid-state electrolytes, 
with in-depth analysis of garnet-type systems and comparative 
analysis across NASICON and perovskite-type families.


## Data
`LiIonDatabase.csv` is included for reproducibility.

Hargreaves, C.J. et al. A database of experimentally measured lithium solid electrolyte conductivities evaluated with machine learning. *npj Computational Materials* **9**, 9 (2023). https://doi.org/10.1038/s41524-022-00951-z

Filter applied: Garnet family, room temperature (15–35°C) >>> 67 samples

Temperature was restricted to minimize thermal effects on ionic conductivity, isolating the influence of composition and dopant type.

Each family was analysed independently to preserve structural 
homogeneity and mechanism-specific interpretability.

Results are then compared across families to assess how compositional features generalise.

## Installation

```bash
# Install dependencies
pip install -r requirements.txt


# Run the full pipeline
python save_dopant.py    # train RF, compute LOOCV, generate SHAP plots
python visualize.py      # additional visualization scripts
python save_nasicon.py   # NASICON family analysis
```



## Method

Parsed chemical compositions into element-wise stoichiometry features using pymatgen

Trained a Random Forest Regressor on log(ionic conductivity)

Applied SHAP (TreeExplainer) to interpret feature contributions

## Model performance

| Family     | LOOCV R² | LOOCV MAE (log₁₀ σ) | 3-fold CV R²  | Ridge baseline   |
|------------|----------|----------------------|---------------|------------------|
| Garnet     | 0.698    | 0.445                | 0.629 ± 0.185 | 0.686 (Train R²) |
| NASICON    | 0.433    | 0.760                | —             | —                |
| Perovskite | 0.272    | 0.696                | —             | —                |

Predictive accuracy decreases with increasing compositional 
diversity. Garnet yields the highest LOOCV R², consistent with 
its simpler substitution chemistry (single Zr site).

NASICON 
and perovskite-type electrolytes show lower accuracy, reflecting 
stronger compositional constraints and greater structural diversity.

Ridge Regression was evaluated as a linear baseline; RF outperformed it (Train R² 0.917 vs 0.686), supporting the use of a non-linear model for this dataset.

LOOCV aggregates leave-one-out predictions across all 67 samples

into a single R², providing a more stable estimate than 3-fold CV

on this small dataset. The Train–LOOCV gap (≈0.22) is expected given N=67 samples paired with ~20 element-wise compositional features.

Combined with the 3-fold CV variance (±0.185, attributable to small fold sizes of ~22 samples),

SHAP feature rankings reported below should be interpreted as preliminary patterns warranting cautious treatment,not definitive conclusions. 

In linear conductivity units, MAE = 0.445 corresponds to a factor of ≈2.8× typical prediction error.

<img width="500" height="511" alt="loocv_parity" src="https://github.com/user-attachments/assets/b3576712-c77e-498f-ae47-7c8be8fdbd38" />

Figure: Garnet LOOCV parity plot. The model captures the overall trend (R²=0.698, MAE=0.445 in log10 σ units).

A mild regression-to-the-mean tendency is visible in the low-conductivity range (log10 σ ≈ −6 to −5), where predictions are systematically slightly higher than actual values — a known behavior on small datasets. 

One notable outlier near log10 σ ≈ −8 corresponds to an unusually low-conductivity garnet variant.




## Results

### Dopant Distribution

<img width="600" height="300" alt="dopant_distribution" src="https://github.com/user-attachments/assets/30a8b57d-6388-450c-b1f6-a4efd8a560e2" />



Significant sample imbalance exists across dopant types (Ta: 32, Nb: 12, Ba: 10, Y: 7, Al: 3).



### SHAP Feature Importance (Bar)

<img width="450" height="595" alt="shap-importance" src="https://github.com/user-attachments/assets/de50a8e5-060f-4e3e-bb53-8aefcaca569b" />

Figure: Mean SHAP value for each element-wise feature. Zr's dominant
magnitude partly reflects its role as a structural proxy — Zr content
varies primarily through B-site substitution by dopants (Ta, Nb, etc.),
so high Zr SHAP captures the substitution gradient rather than an
isolated chemical effect of Zr itself.


### Average SHAP Contribution by Dopant

<img width="560" height="630" alt="dopant_shap_with_reliability" src="https://github.com/user-attachments/assets/5fc31631-927e-432e-b208-b5e7493da53f" />

Figure: Mean SHAP contribution per dopant. Bars with N ≥ 5 (solid)
are statistically supported; bars with N < 5 (faded) are shown for
transparency but their averages are not interpreted due to
small-sample variance. Bar colors indicate sign (blue = positive,
salmon = negative). LLZO backbone elements (Zr, Li, La, O) are
excluded as they are not substitutional dopants.


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

## NASICON Results

The same RF–SHAP pipeline was applied to NASICON-type electrolytes 
(N=154, room temperature 15–35°C) from the Hargreaves database.

LOOCV R²=0.433 (vs. Garnet R²=0.698), reflecting greater compositional 
diversity across substituent types. 

Li and Ti content were excluded from 
the dopant contribution plot as they are structurally coupled to 
substituent amounts (Li = 1+x, Ti = 2−x), which limits per-dopant SHAP 
interpretability.


<img width="700" height="711" alt="nasicon_loocv_parity" src="https://github.com/user-attachments/assets/18ac4516-4419-466a-85de-0a0099b9e743" />


<img width="700" height="750" alt="nasicon_shap-importance" src="https://github.com/user-attachments/assets/b05a5d29-fca0-455b-a79d-6fb09be29736" />

### NASICON SHAP Contribution by Dopant
Among statistically supported dopants (N≥5), Cr and Zr show the
strongest positive mean SHAP contributions. Al shows negative mean
SHAP, likely reflecting multicollinearity with Li content rather
than an isolated dopant effect.

<img width="600" height="720" alt="nasicon_reliability" src="https://github.com/user-attachments/assets/4b963c32-ef51-4911-9754-58cc1e50bc1b" />






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

Among adequately sampled dopants (N ≥ 5: Ta, Nb, Ba, Y, Sr, Sb), Nb and Ba show contrasting mean SHAP contributions, though the difference is small relative to model uncertainty and should be treated as tentative. Ta, despite being the most abundant dopant (N=32) and widely regarded as the most effective LLZO dopant in the literature, shows a slightly negative average (−0.0040) in this dataset — likely reflecting confounding from variable processing conditions not captured by compositional features alone.

Cross-family comparison shows predictive accuracy correlates inversely with compositional diversity (Garnet > NASICON > Perovskite), suggesting composition-based features alone are insufficient for structurally complex families.

Note: Several literature-relevant dopants (Al, Ga, Te, Mg, Hf, etc.) appear in fewer than 5 samples in this filtered subset and are shown in the figure with faded bars for transparency, but their averages are not interpreted due to insufficient sample support.


## Limitations

Garnet subset (N=67) limits generalizability and causes cross-validation instability. 
NASICON (N=154) and Perovskite (N=64) analyses are exploratory and not fully optimised for these families.

Dopant sample imbalance (Ta: 32 vs. Al: 3) may bias SHAP rankings.

Compositional features alone do not capture sintering conditions or grain boundary effects.


## Future Work


 Apply Arrhenius correction to incorporate measurements across full 
 temperature range, expanding usable sample count beyond room-
 temperature subset.

 Incorporate physical descriptors (ionic radius, electronegativity, 
 oxidation state) via matminer to complement element-wise stoichiometry 
 features.

 Design targeted experiments with high-Ta / partial-Zr compositions 
 to disentangle the Zr-Ta confounding identified in this analysis.

 Aggregate larger per-family datasets as more experimental 
 data becomes publicly available.

 NASICON and perovskite analyses revealed limitations of composition-based features under structural constraints, future work should explore physics-informed descriptors (ionic radius, valence, electronegativity) to     
 improve per-family interpretability.





Code: MIT License


Citation: Hargreaves et al., A database of experimentally measured lithium solid electrolyte conductivities evaluated with machine learning, npj Computational Materials, 2023


