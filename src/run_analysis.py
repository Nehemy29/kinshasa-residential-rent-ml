from pathlib import Path
import json
import platform
import warnings

import numpy as np
import pandas as pd
import scipy
from scipy.stats import wilcoxon
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings(
    "ignore",
    message="`sklearn.utils.parallel.delayed` should be used",
    category=UserWarning,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "rent_kinshasa_anonymized.csv"
OUTDIR = ROOT / "results"
OUTDIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(SOURCE)
target = "Loyer_raisonnable"
features = ["Quartier", "Type_logement", "Chambres", "Surface_m2", "Etat", "Eau", "Electricite"]
categorical = ["Quartier", "Type_logement", "Etat", "Eau", "Electricite"]
numeric = ["Chambres", "Surface_m2"]
X = df[features].copy()
y = pd.to_numeric(df[target], errors="raise").to_numpy(float)


def preprocessors():
    scaled = ColumnTransformer([
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric),
    ])
    tree = ColumnTransformer([
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical),
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
        ]), numeric),
    ])
    return scaled, tree


scaled, tree = preprocessors()
models = {
    "Extra Trees": Pipeline([("prep", tree), ("model", ExtraTreesRegressor(
        n_estimators=500, min_samples_leaf=2, max_features=0.8,
        bootstrap=False, random_state=42, n_jobs=1))]),
    "Gradient Boosting": Pipeline([("prep", tree), ("model", GradientBoostingRegressor(
        n_estimators=250, learning_rate=0.03, max_depth=2,
        min_samples_leaf=3, loss="squared_error", random_state=42))]),
    "Forêt aléatoire": Pipeline([("prep", tree), ("model", RandomForestRegressor(
        n_estimators=500, min_samples_leaf=3, max_features=0.8,
        bootstrap=True, random_state=42, n_jobs=1))]),
    "Régression linéaire": Pipeline([("prep", scaled), ("model", LinearRegression())]),
    "Ridge": Pipeline([("prep", scaled), ("model", Ridge(alpha=10.0))]),
}


def grouped_median_predict(train, test, levels):
    train_work = train.copy()
    train_work["_target"] = y[train.index]
    medians = train_work.groupby(levels, dropna=False)["_target"].median()
    global_median = float(np.median(y[train.index]))
    preds = []
    for _, row in test.iterrows():
        key = tuple(row[col] for col in levels)
        if len(levels) == 1:
            key = key[0]
        value = medians.get(key, np.nan)
        if pd.isna(value) and len(levels) > 1:
            fallback = train_work.groupby(levels[0])["_target"].median()
            value = fallback.get(row[levels[0]], np.nan)
        preds.append(global_median if pd.isna(value) else float(value))
    return np.asarray(preds)


splitter = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
splits = list(splitter.split(X))
fold_rows = []
perm_rows = []

for fold, (train_idx, test_idx) in enumerate(splits, start=1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        fold_rows.append({
            "Pli": fold, "Methode": name, "MAE_USD": mean_absolute_error(y_test, pred),
            "RMSE_USD": mean_squared_error(y_test, pred) ** 0.5,
            "R2": r2_score(y_test, pred),
        })
        if name == "Extra Trees":
            perm = permutation_importance(
                model, X_test, y_test, scoring="neg_mean_absolute_error",
                n_repeats=10, random_state=1000 + fold, n_jobs=1,
            )
            for variable, mean_imp in zip(features, perm.importances_mean):
                perm_rows.append({"Pli": fold, "Variable": variable, "Importance_MAE_USD": mean_imp})

    baseline_predictions = {
        "Baseline médiane générale": np.full(len(test_idx), np.median(y_train)),
        "Baseline médiane par quartier": grouped_median_predict(X_train, X_test, ["Quartier"]),
        "Baseline médiane quartier × type": grouped_median_predict(
            X_train, X_test, ["Quartier", "Type_logement"]
        ),
    }
    for name, pred in baseline_predictions.items():
        fold_rows.append({
            "Pli": fold, "Methode": name, "MAE_USD": mean_absolute_error(y_test, pred),
            "RMSE_USD": mean_squared_error(y_test, pred) ** 0.5,
            "R2": r2_score(y_test, pred),
        })

fold_scores = pd.DataFrame(fold_rows)
summary = (fold_scores.groupby("Methode", as_index=False)
           .agg(MAE_moyenne_USD=("MAE_USD", "mean"),
                MAE_ecart_type=("MAE_USD", "std"),
                RMSE_moyenne_USD=("RMSE_USD", "mean"),
                RMSE_ecart_type=("RMSE_USD", "std"),
                R2_moyen=("R2", "mean"),
                R2_ecart_type=("R2", "std"))
           .sort_values("MAE_moyenne_USD"))

perm_df = pd.DataFrame(perm_rows)
perm_summary = (perm_df.groupby("Variable", as_index=False)
                .agg(Importance_moyenne_USD=("Importance_MAE_USD", "mean"),
                     Ecart_type_interplis=("Importance_MAE_USD", "std"),
                     Mediane=("Importance_MAE_USD", "median"),
                     Minimum=("Importance_MAE_USD", "min"),
                     Maximum=("Importance_MAE_USD", "max"))
                .sort_values("Importance_moyenne_USD", ascending=False))

# Fixed out-of-fold predictions for paired, observation-level comparisons.
fixed = KFold(n_splits=5, shuffle=True, random_state=42)
fixed_splits = list(fixed.split(X))
oof_predictions = {name: np.empty(len(df), dtype=float) for name in models}
oof_predictions.update({
    "Baseline médiane générale": np.empty(len(df), dtype=float),
    "Baseline médiane par quartier": np.empty(len(df), dtype=float),
    "Baseline médiane quartier × type": np.empty(len(df), dtype=float),
})

for train_idx, test_idx in fixed_splits:
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train = y[train_idx]
    for name, model in models.items():
        model.fit(X_train, y_train)
        oof_predictions[name][test_idx] = model.predict(X_test)
    oof_predictions["Baseline médiane générale"][test_idx] = np.median(y_train)
    oof_predictions["Baseline médiane par quartier"][test_idx] = grouped_median_predict(
        X_train, X_test, ["Quartier"]
    )
    oof_predictions["Baseline médiane quartier × type"][test_idx] = grouped_median_predict(
        X_train, X_test, ["Quartier", "Type_logement"]
    )

oof_rows = []
for i in range(len(df)):
    row = {"Observation_ID": df.loc[i, "Observation_ID"], "Quartier": df.loc[i, "Quartier"],
           "Loyer_raisonnable": y[i]}
    for name, pred in oof_predictions.items():
        row[f"Prediction__{name}"] = pred[i]
        row[f"Erreur_absolue__{name}"] = abs(y[i] - pred[i])
    oof_rows.append(row)
oof_df = pd.DataFrame(oof_rows)

comparators = [name for name in oof_predictions if name != "Extra Trees"]
test_rows = []
pvals = []
rng = np.random.default_rng(20260926)
extra_err = np.abs(y - oof_predictions["Extra Trees"])
for comparator in comparators:
    comp_err = np.abs(y - oof_predictions[comparator])
    delta = comp_err - extra_err  # positive means Extra Trees is better
    stat, p = wilcoxon(delta, alternative="two-sided", zero_method="pratt")
    bootstrap = np.empty(10000)
    for b in range(len(bootstrap)):
        idx = rng.integers(0, len(delta), len(delta))
        bootstrap[b] = delta[idx].mean()
    test_rows.append({
        "Comparateur": comparator,
        "MAE_Extra_Trees": extra_err.mean(),
        "MAE_Comparateur": comp_err.mean(),
        "Gain_MAE_USD": delta.mean(),
        "Gain_relatif_pct": 100 * delta.mean() / comp_err.mean(),
        "IC95_gain_bas": np.quantile(bootstrap, 0.025),
        "IC95_gain_haut": np.quantile(bootstrap, 0.975),
        "Wilcoxon_W": stat,
        "p_brute": p,
    })
    pvals.append(p)

# Holm adjustment.
order = np.argsort(pvals)
adjusted = np.empty(len(pvals))
running = 0.0
for rank, idx in enumerate(order):
    candidate = min(1.0, (len(pvals) - rank) * pvals[idx])
    running = max(running, candidate)
    adjusted[idx] = running
for row, p_adj in zip(test_rows, adjusted):
    row["p_Holm"] = p_adj
paired_tests = pd.DataFrame(test_rows).sort_values("MAE_Comparateur")

# Leave-one-neighborhood-out evaluation.
lono_rows = []
for held_out in sorted(df["Quartier"].unique()):
    train_idx = df.index[df["Quartier"] != held_out].to_numpy()
    test_idx = df.index[df["Quartier"] == held_out].to_numpy()
    model = models["Extra Trees"]
    model.fit(X.iloc[train_idx], y[train_idx])
    pred = model.predict(X.iloc[test_idx])
    global_pred = np.full(len(test_idx), np.median(y[train_idx]))
    type_pred = grouped_median_predict(X.iloc[train_idx], X.iloc[test_idx], ["Type_logement"])
    for method, values in {
        "Extra Trees": pred,
        "Baseline médiane entraînement": global_pred,
        "Baseline médiane par type": type_pred,
    }.items():
        lono_rows.append({
            "Quartier_exclu": held_out, "Methode": method, "N_test": len(test_idx),
            "MAE_USD": mean_absolute_error(y[test_idx], values),
            "RMSE_USD": mean_squared_error(y[test_idx], values) ** 0.5,
            "R2": r2_score(y[test_idx], values),
            "Biais_moyen_prediction_moins_reel": float(np.mean(values - y[test_idx])),
        })
lono = pd.DataFrame(lono_rows)

metadata = {
    "n_observations": len(df),
    "n_quartiers": int(df["Quartier"].nunique()),
    "validation_principale": "RepeatedKFold 5 plis × 10 répétitions, random_state=42",
    "test_apparie": "Wilcoxon sur erreurs absolues OOF d'un KFold fixe à 5 plis; correction de Holm",
    "bootstrap": "10000 rééchantillonnages des différences appariées de MAE",
    "hyperparametres": {
        "Extra Trees": {"n_estimators": 500, "min_samples_leaf": 2, "max_features": 0.8,
                        "bootstrap": False, "random_state": 42},
        "Gradient Boosting": {"n_estimators": 250, "learning_rate": 0.03, "max_depth": 2,
                              "min_samples_leaf": 3, "loss": "squared_error", "random_state": 42},
        "Forêt aléatoire": {"n_estimators": 500, "min_samples_leaf": 3, "max_features": 0.8,
                            "bootstrap": True, "random_state": 42},
        "Ridge": {"alpha": 10.0},
    },
    "reglage": "Hyperparamètres fixés a priori; aucune recherche sur grille ou optimisation sur cette base.",
    "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                 "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
}

method_names = {
    "Forêt aléatoire": "Random Forest",
    "Régression linéaire": "Linear Regression",
    "Baseline médiane générale": "Overall median",
    "Baseline médiane par quartier": "Neighborhood median",
    "Baseline médiane quartier × type": "Neighborhood-by-housing-type median",
    "Baseline médiane entraînement": "Training-set median",
    "Baseline médiane par type": "Housing-type median",
}

fold_scores_out = fold_scores.rename(columns={
    "Pli": "Fold", "Methode": "Method", "MAE_USD": "MAE_USD",
    "RMSE_USD": "RMSE_USD", "R2": "R2",
})
fold_scores_out["Method"] = fold_scores_out["Method"].replace(method_names)

summary_out = summary.rename(columns={
    "Methode": "Method", "MAE_moyenne_USD": "Mean_MAE_USD",
    "MAE_ecart_type": "MAE_SD", "RMSE_moyenne_USD": "Mean_RMSE_USD",
    "RMSE_ecart_type": "RMSE_SD", "R2_moyen": "Mean_R2",
    "R2_ecart_type": "R2_SD",
})
summary_out["Method"] = summary_out["Method"].replace(method_names)

perm_out = perm_df.rename(columns={
    "Pli": "Fold", "Variable": "Feature", "Importance_MAE_USD": "MAE_importance_USD"
})
perm_summary_out = perm_summary.rename(columns={
    "Variable": "Feature", "Importance_moyenne_USD": "Mean_importance_USD",
    "Ecart_type_interplis": "Fold_SD", "Mediane": "Median",
    "Minimum": "Minimum", "Maximum": "Maximum",
})

paired_out = paired_tests.rename(columns={
    "Comparateur": "Comparator", "MAE_Extra_Trees": "Extra_Trees_MAE",
    "MAE_Comparateur": "Comparator_MAE", "Gain_MAE_USD": "MAE_gain_USD",
    "Gain_relatif_pct": "Relative_gain_pct", "IC95_gain_bas": "CI95_lower",
    "IC95_gain_haut": "CI95_upper", "p_brute": "Raw_p", "p_Holm": "Holm_p",
})
paired_out["Comparator"] = paired_out["Comparator"].replace(method_names)

lono_out = lono.rename(columns={
    "Quartier_exclu": "Held_out_neighborhood", "Methode": "Method",
    "N_test": "N_test", "MAE_USD": "MAE_USD", "RMSE_USD": "RMSE_USD",
    "R2": "R2", "Biais_moyen_prediction_moins_reel": "Mean_prediction_bias",
})
lono_out["Method"] = lono_out["Method"].replace(method_names)

fold_scores_out.to_csv(OUTDIR / "fold_scores.csv", index=False)
summary_out.to_csv(OUTDIR / "model_and_baseline_summary.csv", index=False)
perm_out.to_csv(OUTDIR / "fold_permutation_importance.csv", index=False)
perm_summary_out.to_csv(OUTDIR / "permutation_importance_summary.csv", index=False)
oof_df.to_csv(OUTDIR / "out_of_fold_predictions.csv", index=False)
paired_out.to_csv(OUTDIR / "paired_tests.csv", index=False)
lono_out.to_csv(OUTDIR / "leave_one_neighborhood_out.csv", index=False)
(OUTDIR / "metadata.json").write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
)

print(summary_out.to_string(index=False))
print("\nIMPORTANCES\n", perm_summary.to_string(index=False))
print("\nPAIRED TESTS\n", paired_out.to_string(index=False))
print("\nLEAVE-ONE-NEIGHBORHOOD-OUT\n", lono_out.to_string(index=False))
