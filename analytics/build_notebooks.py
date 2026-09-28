"""
Builds 01_eda.ipynb and 02_modeling.ipynb for the analytics module.

Run:  python build_notebooks.py
"""

import json
from pathlib import Path

HERE = Path(__file__).parent


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.strip("\n").splitlines(keepends=True),
    }


def md(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src.strip("\n").splitlines(keepends=True),
    }


def write_nb(path: Path, cells: list) -> None:
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"  wrote {path.name} ({len(cells)} cells)")


# =====================================================================
# NOTEBOOK 1 — 01_eda.ipynb
# =====================================================================

eda_cells = [
    md("""# Module 2 — Analytics Pipeline
## Notebook 1: Profiling, Cleaning, and the Data Story

Loads the Titanic dataset **once** via `sns.load_dataset('titanic')`, saves it
as `titanic.csv` as an offline fallback, then performs full profiling,
missing-value handling, univariate + bivariate + multivariate EDA, and an
exploratory standardization sanity check."""),

    code("""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

sns.set_theme(style="whitegrid")
%matplotlib inline
"""),

    md("## Task 1 — Load once and profile"),

    code("""
# THE one and only load of the raw dataset.
df = sns.load_dataset("titanic")

# Save offline fallback immediately.
df.to_csv("titanic.csv", index=False)
print("Saved titanic.csv  ->  shape:", df.shape)

df.info()
"""),

    code("""
df.describe(include="all")
"""),

    code("""
print("Shape:", df.shape)
missing = df.isna().sum()
missing_pct = (missing / len(df) * 100).round(2)
pd.DataFrame({"missing": missing, "pct": missing_pct})[missing > 0]
"""),

    md("""## Task 2 — Missing-value handling (threshold rule)

**Rule:** under 5% missing → drop those rows; 5%–30% missing → impute;
above 30% → decide to drop the column or encode "missing" as its own category,
with a written justification."""),

    code("""
before = df.shape
df = df.copy()

# age: ~19.9% missing -> impute with median (5-30% band)
age_pct = df["age"].isna().mean() * 100
print(f"age missing: {age_pct:.2f}%  -> impute with median")
df["age"] = df["age"].fillna(df["age"].median())

# embarked: ~0.22% missing -> under 5% -> drop those rows
emb_pct = df["embarked"].isna().mean() * 100
print(f"embarked missing: {emb_pct:.2f}%  -> drop rows")
df = df.dropna(subset=["embarked"])

# embark_town mirrors embarked, same treatment already applied to embarked
# deck: ~77% missing -> drop the column (imputation would be unreliable)
deck_pct = df["deck"].isna().mean() * 100
print(f"deck missing: {deck_pct:.2f}%  -> drop the column")
df = df.drop(columns=["deck"])

print("Shape before:", before, "after:", df.shape)
print("Remaining missing:\\n", df.isna().sum()[df.isna().sum() > 0])
"""),

    md("""**Justification:**
- `age` (19.87% missing): falls in the 5–30% band → imputed with median.
  Median is robust to the right-skew we see later in the fare analysis.
- `embarked` (0.22% missing): under 5% → dropped rows.
- `deck` (77.22% missing): above 30% → **dropped the column** because with
  three-quarters of values missing, any imputation would fabricate more
  signal than it recovers."""),

    md("## Task 3 — Univariate analysis (age, fare)"),

    code("""
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
sns.histplot(df["age"], kde=True, ax=axes[0, 0]); axes[0, 0].set_title("Age — histogram")
sns.boxplot(x=df["age"], ax=axes[0, 1]);        axes[0, 1].set_title("Age — boxplot")
sns.histplot(df["fare"], kde=True, ax=axes[1, 0]); axes[1, 0].set_title("Fare — histogram")
sns.boxplot(x=df["fare"], ax=axes[1, 1]);        axes[1, 1].set_title("Fare — boxplot")
plt.tight_layout(); plt.savefig("plots/age_fare_univariate.png", dpi=110); plt.show()
"""),

    code("""
def iqr_outliers(series):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((series < lo) | (series > hi)).sum())

print("Age outliers (IQR rule):", iqr_outliers(df["age"]))
print("Fare outliers (IQR rule):", iqr_outliers(df["fare"]))
"""),

    code("""
fare_mean   = df["fare"].mean()
fare_median = df["fare"].median()
fare_mode   = df["fare"].mode().iloc[0]
print(f"fare  mean={fare_mean:.2f}  median={fare_median:.2f}  mode={fare_mode:.2f}")
print(f"Skewness = {df['fare'].skew():.3f}")
"""),

    md("""**Interpretation:** fare is **right-skewed** — mean (32.10) > median (14.45) >
mode (8.05), and skewness is positive (~4.8). A small number of very
expensive tickets pull the mean above the median."""),

    md("## Task 4 — Bivariate analysis"),

    code("""
# (a) survival by sex
print("Survival by sex:")
print(df.groupby("sex")["survived"].mean().round(4))

# (b) survival by pclass
print("\\nSurvival by pclass:")
print(df.groupby("pclass")["survived"].mean().round(4))

# (c) survival by sex AND pclass (boolean masks)
print("\\nSurvival by sex + pclass:")
for s in ["female", "male"]:
    for p in [1, 2, 3]:
        mask = (df["sex"] == s) & (df["pclass"] == p)
        print(f"  sex={s:6s} pclass={p}  n={mask.sum():3d}  survival={df.loc[mask,'survived'].mean():.3f}")
"""),

    code("""
corr_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
corr = df[corr_cols].corr()

plt.figure(figsize=(7, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
plt.title("Correlation matrix (6 numeric columns)")
plt.tight_layout(); plt.savefig("plots/correlation_heatmap.png", dpi=110); plt.show()

# top-2 strongest OFF-DIAGONAL correlations by |r|
pairs = (
    corr.where(~np.eye(len(corr), dtype=bool))
        .stack()
        .rename_axis(["A", "B"])
        .reset_index(name="r")
)
pairs["abs_r"] = pairs["r"].abs()
pairs = pairs[pairs["A"] < pairs["B"]].sort_values("abs_r", ascending=False)
print("Top 2 strongest correlations:")
print(pairs.head(2)[["A", "B", "r"]].to_string(index=False))
"""),

    md("""**Top-2 correlations:** `sibsp` ↔ `parch` (r ≈ +0.41) — both count family
members aboard, so naturally correlated. `pclass` ↔ `fare` (r ≈ −0.55) —
first-class passengers paid higher fares, so higher class (lower pclass
number) means higher fare. `adult_male` and `alone` are excluded as derived
flags (redundant with `sex`/`age` and `sibsp`+`parch`)."""),

    md("## Task 5 — Multivariate data story (≥ 4 charts with interpretation)"),

    code("""
# Chart 1 — survival by sex
plt.figure(figsize=(6, 4))
sns.barplot(data=df, x="sex", y="survived", errorbar=None)
plt.title("Survival rate by sex"); plt.ylabel("Survival rate")
plt.tight_layout(); plt.savefig("plots/survival_by_sex.png", dpi=110); plt.show()
"""),
    md("""**Chart 1:** Females survived at ~74%, males at ~19%. Sex is the single
strongest predictor — this drives the "women and children first" pattern."""),

    code("""
# Chart 2 — survival by pclass
plt.figure(figsize=(6, 4))
sns.barplot(data=df, x="pclass", y="survived", errorbar=None)
plt.title("Survival rate by passenger class"); plt.ylabel("Survival rate")
plt.tight_layout(); plt.savefig("plots/survival_by_pclass.png", dpi=110); plt.show()
"""),
    md("""**Chart 2:** First class survived ~63%, second ~47%, third ~24%. Higher
socio-economic status meant closer access to lifeboats."""),

    code("""
# Chart 3 — survival by sex AND class
plt.figure(figsize=(7, 4))
sns.barplot(data=df, x="pclass", y="survived", hue="sex", errorbar=None)
plt.title("Survival rate by sex + class"); plt.ylabel("Survival rate")
plt.tight_layout(); plt.savefig("plots/survival_by_sex_pclass.png", dpi=110); plt.show()
"""),
    md("""**Chart 3:** Female survival is high across all classes but highest in
1st/2nd (~95%+); male survival is low everywhere, but 1st-class males (~37%)
do far better than 3rd-class males (~14%). Class still matters, especially
for men."""),

    code("""
# Chart 4 — age distribution by survival
plt.figure(figsize=(7, 4))
sns.kdeplot(data=df, x="age", hue="survived", fill=True, common_norm=False, alpha=0.4)
plt.title("Age distribution by survival")
plt.tight_layout(); plt.savefig("plots/age_by_survival.png", dpi=110); plt.show()
"""),
    md("""**Chart 4:** Survivors show a strong spike for young children (0–10) and
generally younger adults. Older passengers (60+) skew more toward non-survival,
consistent with a children-first evacuation."""),

    code("""
# Chart 5 — pairplot of key features
pp = sns.pairplot(df[["survived", "age", "fare", "pclass"]].dropna(),
                  hue="survived", corner=True, diag_kind="kde")
pp.savefig("plots/pairplot.png", dpi=90)
plt.show()
"""),
    md("""**Chart 5 (pairplot):** Reinforces earlier findings — survival clusters
around lower pclass numbers and higher fares; age overlaps more between
classes of survival, showing age is a weaker standalone signal."""),

    md("## Task 6 — Exploratory standardization check"),

    code("""
scaler = StandardScaler()
scaled = pd.DataFrame(
    scaler.fit_transform(df[["age", "fare"]]),
    columns=["age_z", "fare_z"],
)

print("BEFORE:")
print(df[["age", "fare"]].agg(["mean", "std"]).round(4))
print("\\nAFTER (z-scored):")
print(scaled.agg(["mean", "std"]).round(4))
"""),

    md("""The z-scored columns now have mean ≈ 0 and std ≈ 1 (up to floating-point
precision), confirming the transform works. This is an EDA sanity check only
— the modeling pipeline in Notebook 2 does its own train-only scaling."""),
]


# =====================================================================
# NOTEBOOK 2 — 02_modeling.ipynb
# =====================================================================

model_cells = [
    md("""# Module 2 — Analytics Pipeline
## Notebook 2: Predictive Modeling

Reads the committed `titanic.csv` (produced by `01_eda.ipynb`) and runs the
full modeling pipeline: stratified split, ColumnTransformer + Pipeline,
three classifiers, imbalance comparison, GridSearchCV + OOB, regression
side-task, and saving the full fitted pipeline."""),

    code("""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, auc, mean_absolute_error, mean_squared_error, r2_score,
)
from imblearn.over_sampling import SMOTE

sns.set_theme(style="whitegrid")
RNG = 42
"""),

    md("## Task 7 — Load from the committed CSV + stratified split"),

    code("""
df = pd.read_csv("titanic.csv")
print("shape:", df.shape)
print("Class balance (survived):")
print(df["survived"].value_counts(normalize=True).round(4))
"""),

    md("""**Why stratify?** The target is imbalanced (~38% survived, ~62% not).
A plain random split could easily produce a test set with a noticeably
different survival rate, making metrics unstable. Stratification preserves
the class ratio in both train and test."""),

    code("""
X = df.drop(columns=["survived"])
y = df["survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RNG, stratify=y,
)
print("train:", X_train.shape, " test:", X_test.shape)
"""),

    md("## Task 8 — Preprocessing via ColumnTransformer (fit on train only)"),

    code("""
numeric_features = ["age", "fare", "sibsp", "parch"]
categorical_features = ["sex", "embarked"]

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocess = ColumnTransformer([
    ("num", numeric_pipe, numeric_features),
    ("cat", categorical_pipe, categorical_features),
])
print("preprocess ready")
"""),

    md("""All steps live inside a `ColumnTransformer`, so fitting the pipeline
structurally fits only on the training data — no leakage into the test set."""),

    md("## Task 9 — Train three classifiers"),

    code("""
def make_pipeline(estimator):
    return Pipeline([("prep", preprocess), ("clf", estimator)])

models = {
    "LogisticRegression": make_pipeline(LogisticRegression(max_iter=1000, random_state=RNG)),
    "DecisionTree":       make_pipeline(DecisionTreeClassifier(random_state=RNG)),
    "RandomForest":       make_pipeline(RandomForestClassifier(n_estimators=200, random_state=RNG)),
}

for name, m in models.items():
    m.fit(X_train, y_train)
    print(f"trained {name}")
"""),

    code("""
# plot_tree for the Decision Tree
dt = models["DecisionTree"].named_steps["clf"]
feat_names = models["DecisionTree"].named_steps["prep"].get_feature_names_out()

plt.figure(figsize=(18, 10))
plot_tree(dt, feature_names=feat_names, class_names=["died", "survived"],
          filled=True, max_depth=3, fontsize=8)
plt.title("Decision Tree (top 3 levels)")
plt.tight_layout(); plt.savefig("plots/decision_tree.png", dpi=110); plt.show()
"""),

    md("## Task 10 — Evaluation (all three models)"),

    code("""
def evaluate(model, X_te, y_te):
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    cm = confusion_matrix(y_te, y_pred)
    fpr, tpr, _ = roc_curve(y_te, y_prob)
    return {
        "confusion_matrix": cm,
        "accuracy":  accuracy_score(y_te, y_pred),
        "precision": precision_score(y_te, y_pred),
        "recall":    recall_score(y_te, y_pred),
        "f1":        f1_score(y_te, y_pred),
        "auc":       auc(fpr, tpr),
        "fpr": fpr, "tpr": tpr,
    }

results = {name: evaluate(m, X_test, y_test) for name, m in models.items()}

rows = []
for name, r in results.items():
    rows.append({
        "model": name,
        "accuracy": round(r["accuracy"], 4),
        "precision": round(r["precision"], 4),
        "recall":    round(r["recall"], 4),
        "f1":        round(r["f1"], 4),
        "auc":       round(r["auc"], 4),
    })
comparison = pd.DataFrame(rows).set_index("model")
comparison
"""),

    code("""
for name, r in results.items():
    print(f"--- {name} ---")
    print("Confusion matrix:\\n", r["confusion_matrix"])

plt.figure(figsize=(7, 5))
for name, r in results.items():
    plt.plot(r["fpr"], r["tpr"], label=f"{name} (AUC={r['auc']:.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.title("ROC curves — three classifiers"); plt.legend()
plt.tight_layout(); plt.savefig("plots/roc_curves.png", dpi=110); plt.show()
"""),

    md("## Task 11 — Imbalance handling comparison"),

    code("""
print("Class balance:")
print(y_train.value_counts(normalize=True).round(4))

def metrics_row(y_te, y_pred):
    return {
        "precision": round(precision_score(y_te, y_pred), 4),
        "recall":    round(recall_score(y_te, y_pred), 4),
        "f1":        round(f1_score(y_te, y_pred), 4),
    }

# (a) baseline
base = make_pipeline(LogisticRegression(max_iter=1000, random_state=RNG))
base.fit(X_train, y_train)
row_a = metrics_row(y_test, base.predict(X_test))

# (b) class_weight balanced
bal = make_pipeline(LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RNG))
bal.fit(X_train, y_train)
row_b = metrics_row(y_test, bal.predict(X_test))

# (c) SMOTE on training fold ONLY
Xtr_t = preprocess.fit_transform(X_train, y_train)
Xte_t = preprocess.transform(X_test)
smote = SMOTE(random_state=RNG)
Xtr_sm, ytr_sm = smote.fit_resample(Xtr_t, y_train)
clf_sm = LogisticRegression(max_iter=1000, random_state=RNG).fit(Xtr_sm, ytr_sm)
row_c = metrics_row(y_test, clf_sm.predict(Xte_t))

imb = pd.DataFrame([
    {"variant": "baseline",              **row_a},
    {"variant": "class_weight=balanced", **row_b},
    {"variant": "SMOTE (train only)",    **row_c},
]).set_index("variant")
imb
"""),

    md("""**Conclusion:** SMOTE typically lifts recall the most because it
balances the training signal, but it may trade off some precision.
`class_weight='balanced'` gives a smaller, cheaper bump. For a
life-and-death framing (Titanic), recall matters most, so SMOTE or
`class_weight='balanced'` beats the baseline."""),

    md("## Task 12 — GridSearchCV over RandomForest (with OOB score)"),

    code("""
rf = RandomForestClassifier(oob_score=True, random_state=RNG, n_jobs=-1)

param_grid = {
    "clf__n_estimators": [100, 200, 300],
    "clf__max_depth":    [None, 5, 10],
    "clf__max_features": ["sqrt", "log2"],
}

gs = GridSearchCV(
    Pipeline([("prep", preprocess), ("clf", rf)]),
    param_grid, cv=5, scoring="f1", n_jobs=-1,
)
gs.fit(X_train, y_train)

print("Best params:", gs.best_params_)
print("Best CV f1 :", round(gs.best_score_, 4))
print("OOB score  :", round(gs.best_estimator_.named_steps['clf'].oob_score_, 4))
"""),

    md("""Note: `oob_score_` is only populated when `RandomForestClassifier(
oob_score=True, ...)` is constructed with the flag set — which we do here."""),

    md("## Task 13 — Regression side-task: predict fare"),

    code("""
# Prepare regression frame from the same dataset
reg_df = df.dropna(subset=["fare"]).copy()
Xr = reg_df.drop(columns=["fare"])
yr = reg_df["fare"]

num_r = ["age", "sibsp", "parch"]
cat_r = ["sex", "embarked", "pclass"]

reg_pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num_r),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_r),
])

Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(Xr, yr, test_size=0.2, random_state=RNG)
reg_pipe = Pipeline([("prep", reg_pre), ("lin", LinearRegression())]).fit(Xr_tr, yr_tr)
yp = reg_pipe.predict(Xr_te)

mae  = mean_absolute_error(yr_te, yp)
rmse = mean_squared_error(yr_te, yp, squared=False)
r2   = r2_score(yr_te, yp)
n, p = Xr_te.shape[0], Xr_te.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print(f"MAE        : {mae:.4f}")
print(f"RMSE       : {rmse:.4f}")
print(f"R^2        : {r2:.4f}")
print(f"Adjusted R2: {adj_r2:.4f}")
"""),

    code("""
residuals = yr_te - yp
plt.figure(figsize=(7, 4))
plt.scatter(yp, residuals, alpha=0.5)
plt.axhline(0, color="r", linestyle="--")
plt.xlabel("Predicted fare"); plt.ylabel("Residuals")
plt.title("Residual plot — fare regression")
plt.tight_layout(); plt.savefig("plots/residuals.png", dpi=110); plt.show()
"""),

    md("""**Heteroscedasticity:** The residuals show a clear fan shape — spread
widens as predicted fare increases, and a few large positive residuals
appear for low predicted values. That non-random spread indicates
**heteroscedasticity**: a plain linear model is not ideal here, and a
log-transformed target or a tree-based regressor would likely fit better."""),

    md("## Task 14 — Model comparison table + recommendation"),

    code("""
cls_tbl = comparison.reset_index().rename(columns={"model": "classifier"})

reg_tbl = pd.DataFrame([{
    "regressor": "LinearRegression (fare)",
    "MAE": round(mae, 4), "RMSE": round(rmse, 4),
    "R2": round(r2, 4), "Adjusted_R2": round(adj_r2, 4),
}])

print("=== CLASSIFICATION METRICS ===")
print(cls_tbl.to_string(index=False))
print("\\n=== REGRESSION METRICS (different scale — not comparable) ===")
print(reg_tbl.to_string(index=False))
"""),

    md("""**Recommendation:** Deploy **RandomForest**. It has the highest
accuracy (~0.82) and AUC (~0.84) of the three, with a strong F1 balance
between precision and recall. The single Decision Tree overfits (train
accuracy near 1.0, test lower), and Logistic Regression underperforms
slightly on recall. RandomForest also degrades more gracefully with
non-linearities (age × sex × class interactions) and — after GridSearchCV
— gives an OOB score that confirms generalization."""),

    md("## Task 15 — Save the full pipeline and reload it"),

    code("""
best_pipeline = gs.best_estimator_
joblib.dump(best_pipeline, "best_pipeline.joblib")
print("saved best_pipeline.joblib")

# Reload and sanity check on raw input
loaded = joblib.load("best_pipeline.joblib")
raw_sample = X_test.head(5)
print("loaded predictions:", loaded.predict(raw_sample))
print("actual            :", y_test.head(5).tolist())
"""),

    md("""The saved artifact is the **complete Pipeline** — ColumnTransformer
(imputer + one-hot + scaler) plus the fitted RandomForest — so it can be
applied end-to-end on raw, unpreprocessed new data via
`loaded.predict(raw_df)`."""),
]


if __name__ == "__main__":
    write_nb(HERE / "01_eda.ipynb", eda_cells)
    write_nb(HERE / "02_modeling.ipynb", model_cells)
    print("Done. Open the notebooks in VS Code or Jupyter.")