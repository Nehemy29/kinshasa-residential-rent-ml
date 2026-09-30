# Machine Learning-Based Residential Rent Modeling in Kinshasa

This repository contains the reproducible analysis supporting the manuscript
**“Machine Learning-Based Predictive Modeling of Perceived Reasonable Residential Rent in Kinshasa.”**

The study compares five regression algorithms and three naive median baselines using 150 anonymized housing observations from Mombele, Kingabwa, and Limete Résidentiel. Extra Trees is the primary model. The repository also reproduces paired statistical tests, validation-fold permutation importance, and leave-one-neighborhood-out validation.

## Main findings reproduced by the code

- Extra Trees repeated-cross-validation MAE: approximately **USD 11.87**.
- Extra Trees repeated-cross-validation R²: approximately **0.993**.
- Extra Trees fixed out-of-fold MAE: approximately **USD 10.56**.
- Strongest naive baseline out-of-fold MAE: approximately **USD 44.00**.
- Relative MAE reduction against that baseline: approximately **76.0%**.
- Leave-one-neighborhood-out Extra Trees MAE: **USD 82.99–261.55**, indicating weak geographic transferability.

## Repository structure

```text
.
├── data/
│   ├── README.md
│   └── rent_kinshasa_anonymized.csv
├── results/
│   └── .gitkeep
├── src/
│   ├── run_analysis.py
│   └── validate_reproduction.py
├── .gitignore
├── CITATION.cff
├── LICENSE
├── README.md
└── requirements.txt
```

## Installation

Python 3.12 is recommended.

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On Windows Command Prompt:

```cmd
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

## Reproduce the analysis

From the repository root:

```bash
python src/run_analysis.py
python src/validate_reproduction.py
```

The first command writes all output tables to `results/`. The second checks the principal values reported in the manuscript using small numerical tolerances.

## Methodological safeguards

- Asking rent and agreed rent are excluded from model predictors to reduce circularity and leakage.
- All preprocessing steps are fitted inside each validation fold.
- Median baselines are calculated exclusively from training folds.
- Permutation importance is calculated on validation observations, not training observations.
- Hyperparameters are fixed a priori; no grid search is performed on the 150 observations.
- Geographic transferability is tested by excluding each neighborhood completely from training.

## Interpretation warning

The target is **perceived reasonable rent**, not a legal, administrative, causal, or objective market value. The dataset covers only three neighborhoods in one municipality. High internal accuracy must not be interpreted as citywide validity. The leave-one-neighborhood-out results show that broader territorial data are required before operational deployment across Kinshasa.

## Data governance

The included analytical dataset does not contain names, telephone numbers, or exact personal addresses. Before making the repository public, the author should complete a final review of the original consent conditions and collection metadata. See `data/README.md`.

## Citation

Citation metadata are provided in `CITATION.cff`. Update the manuscript DOI, journal details, repository URL, and release date after publication.

## License

The analysis code is released under the MIT License. The dataset should only be redistributed after confirmation that the original consent conditions authorize public sharing.

