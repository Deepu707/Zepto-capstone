# Module 2 — Analytics Pipeline

End-to-end analytics workflow on the classic Titanic dataset: profile → clean
→ visual story → predict → evaluate → deploy. Split into two notebooks that
share one committed CSV.

---

## Deliverables in this folder

| File | Purpose |
|---|---|
| `01_eda.ipynb` | Loads data once, profiles, cleans, saves `titanic.csv`, produces EDA + charts |
| `02_modeling.ipynb` | Reads `titanic.csv`, builds + evaluates 3 classifiers, regression side-task, saves pipeline |
| `titanic.csv` | Offline fallback of the raw dataset (committed, produced by `01_eda.ipynb`) |
| `best_pipeline.joblib` | Full fitted `Pipeline` (preprocessing + RandomForest) |
| `build_notebooks.py` | Generates both notebooks from source (kept for reproducibility) |
| `requirements.txt` | Python dependencies |
| `plots/*.png` | Supporting chart images (never a substitute for written interpretation) |

---

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

> **First run of `01_eda.ipynb` needs internet**: `sns.load_dataset('titanic')`
> fetches the dataset from Seaborn's online repo and caches it. After that,
> the committed `titanic.csv` is the sole offline fallback used by
> `02_modeling.ipynb`.

---

## How to run

1. Open `01_eda.ipynb`, select the `(venv)` kernel, **Run All**.
   → saves `titanic.csv`, all EDA charts to `plots/`, prints profiling numbers.
2. Open `02_modeling.ipynb`, **Run All**.
   → reads `titanic.csv`, trains 3 classifiers, runs GridSearchCV, fits the
   regression side-task, and saves `best_pipeline.joblib`.

The raw dataset is loaded from network/cache **exactly once** (Notebook 1).
Notebook 2 uses only `pd.read_csv("titanic.csv")`.

---

## Part A — EDA findings (Notebook 1)

### Task 1 — Profiling
- Loaded via `sns.load_dataset("titanic")`; shape **(891, 15)**; saved as `titanic.csv`.
- Columns with missing values: `age` (19.87%), `embarked` (0.22%),
  `deck` (77.22%), `embark_town` (0.22%).

### Task 2 — Missing-value handling (threshold rule)

| Column | Missing % | Rule applied | Justification |
|---|---|---|---|
| `age` | 19.87% | **Impute with median** | In 5–30% band; median robust to right-skew |
| `embarked` | 0.22% | **Drop rows** | Under 5% |
| `deck` | 77.22% | **Drop the column** | Over 30%; imputation would fabricate signal |
| `embark_town` | 0.22% | Drop rows (mirrors `embarked`) | Under 5% |

### Task 3 — Univariate (age, fare)
- **Age** IQR outliers: **65**. **Fare** IQR outliers: **114**.
- Fare statistics: **mean ≈ 32.10**, **median ≈ 14.45**, **mode ≈ 8.05**,
  skewness ≈ **+4.79**.
- **Conclusion:** fare is **right-skewed** (mean > median > mode).

### Task 4 — Bivariate survival
- **By sex:** female ≈ **0.742**, male ≈ **0.189**.
- **By pclass:** 1 ≈ **0.630**, 2 ≈ **0.473**, 3 ≈ **0.242**.
- **By sex + pclass** (survival rate): females 1st ≈ 0.968, 2nd ≈ 0.921,
  3rd ≈ 0.500; males 1st ≈ 0.369, 2nd ≈ 0.157, 3rd ≈ 0.135.
- **Correlation matrix** on exactly `["survived", "pclass", "age", "sibsp", "parch", "fare"]`;
  `adult_male` and `alone` excluded as derived/redundant flags.
- **Top-2 strongest off-diagonal correlations:**
  1. `sibsp` ↔ `parch` ≈ **+0.41** — both count family aboard.
  2. `pclass` ↔ `fare` ≈ **−0.55** — higher class means higher fare.

### Task 5 — Multivariate data story (5 charts, each with interpretation)

1. **Survival by sex** — females ~74%, males ~19%. Sex is the strongest single signal.
2. **Survival by class** — 1st ~63%, 2nd ~47%, 3rd ~24%. Class independently mattered.
3. **Survival by sex + class** — 1st/2nd-class females nearly certain to survive
   (~92–97%); 3rd-class males worst (~14%).
4. **Age distribution by survival** — young children spike in survivors;
   older passengers (60+) skew non-survivor.
5. **Pairplot (survived, age, fare, pclass)** — reinforces the class/fare split;
   age alone separates less cleanly.

Each chart is saved to `plots/` and paired with its own 2–4 sentence interpretation
in notebook Markdown cells (and summarized above).

### Task 6 — Standardization sanity check
- Z-scored `age` and `fare` (exploratory only, does not feed the modeling pipeline).
- Before: age mean 29.3152, std 12.9849; fare mean 32.0967, std 49.6975.
- After: age_z mean 0.0000, std 1.0006; fare_z mean 0.0000, std 1.0006.
  (std = 1.0006, not exactly 1, because pandas `.std()` uses `ddof=1` while
  `StandardScaler` uses `ddof=0` — expected.)

---

## Part B — Modeling (Notebook 2)

### Task 7 — Split

- **Class balance:** survived=0 → **0.6162**, survived=1 → **0.3838**.
- **Stratified split (80/20, random_state=42).**
- **Why stratify:** the ~62/38 class imbalance means a random split could
  easily produce a test set with a different survival rate, destabilizing
  precision/recall/F1. Stratification preserves the class ratio in both
  train and test.

### Task 8 — Preprocessing (fit on training data only)

- Numeric (`age`, `fare`, `sibsp`, `parch`): `SimpleImputer(median)` → `StandardScaler`.
- Categorical (`sex`, `embarked`): `SimpleImputer(most_frequent)` → `OneHotEncoder`.
- All wrapped in a `ColumnTransformer` inside a `Pipeline`, so **fit-on-train
  / transform-only on test** is structurally enforced — no leakage.

### Task 9 — Three classifiers
Logistic Regression, Decision Tree (also rendered with `plot_tree`, top-3
levels, labeled features + classes), and Random Forest (`n_estimators=200`).

### Task 10 — Metrics (side by side)

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.7989 | 0.7797 | 0.6667 | 0.7188 | 0.8194 |
| Decision Tree | 0.7709 | 0.7059 | 0.6957 | 0.7007 | 0.7514 |
| Random Forest | 0.7877 | 0.7627 | 0.6522 | 0.7031 | **0.8239** |

Confusion matrices, ROC curves for all three (saved as `plots/roc_curves.png`),
and a full metric suite are all included in the notebook.

### Task 11 — Imbalance handling comparison

| Variant | Precision | Recall | F1 |
|---|---|---|---|
| Baseline (no handling) | **0.7797** | 0.6667 | **0.7188** |
| `class_weight='balanced'` | 0.7385 | 0.6957 | 0.7164 |
| SMOTE (train fold only) | 0.7385 | 0.6957 | 0.7164 |

**Conclusion.** On this dataset, the three variants are effectively tied on F1
(0.7188 vs 0.7164). The baseline keeps the **highest precision** but the lowest
recall; both `class_weight='balanced'` and SMOTE trade a small amount of
precision for ~3 points of recall, which is a favorable trade in a
safety-critical framing (missing a survivor is worse than a false alarm).
SMOTE was applied **only to the training fold** via `fit_resample` on the
already-split training data — no leakage into test. **Recommendation for
imbalanced use-cases here: `class_weight='balanced'`**, which delivers the
same recall gain as SMOTE at zero preprocessing overhead.

### Task 12 — GridSearchCV (RandomForest, `oob_score=True`)

- **Best params:** `{'clf__max_depth': 5, 'clf__max_features': 'sqrt', 'clf__n_estimators': 100}`
- **Best CV F1:** **0.749**
- **OOB score:** **0.8146** — confirms the tuned forest generalizes well.

### Task 13 — Regression side-task (predict `fare`)

| Metric | Value |
|---|---|
| MAE | **18.8702** |
| RMSE | **30.9202** |
| R² | **0.3822** |
| Adjusted R² | **0.3294** |

**Heteroscedasticity:** the residual plot (saved as `plots/residuals.png`)
shows a clear fan pattern — residuals widen as predicted fare increases, and
large positive residuals cluster at low predicted values. This is
**heteroscedasticity**: the linear model does not fit fare's variance well.
A log-transformed target or a tree-based regressor would likely improve it.

### Task 14 — Model comparison table

**Classification metrics (one metric group):**

| Classifier | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.7989 | 0.7797 | 0.6667 | 0.7188 | 0.8194 |
| Decision Tree | 0.7709 | 0.7059 | 0.6957 | 0.7007 | 0.7514 |
| Random Forest | 0.7877 | 0.7627 | 0.6522 | 0.7031 | **0.8239** |

**Regression metrics (separate metric group — different scale, not comparable):**

| Regressor | MAE | RMSE | R² | Adjusted R² |
|---|---|---|---|---|
| Linear Regression (fare) | 18.8702 | 30.9202 | 0.3822 | 0.3294 |

> Classification metrics (0–1 probabilities) and regression metrics
> (currency-scale errors) live on **different scales** and are presented
> as **two distinct metric groups**, never merged into one column.

**Recommendation.** Deploy the **Random Forest**. It has the best AUC
(0.8239) and strong accuracy (0.7877), and its precision/recall trade-off
sits between LogReg and the single Decision Tree. The Decision Tree is the
weakest (accuracy 0.7709, AUC 0.7514) and, being a single tree, overfits
more readily. GridSearchCV + OOB = 0.8146 confirms the Random Forest
generalizes beyond the test set. If a **recall-first** objective were
required (e.g., prioritizing survivors), LogReg with
`class_weight='balanced'` would be a close second choice.

### Task 15 — Saved pipeline

```python
joblib.dump(gs.best_estimator_, "best_pipeline.joblib")
```

The saved artifact is the **complete `Pipeline`** (ColumnTransformer +
RandomForest), so it can be applied to raw, unpreprocessed new data
end-to-end. Reload sanity check:

```python
loaded = joblib.load("best_pipeline.joblib")
loaded.predict(X_test.head(5))   # -> [0 0 0 0 1]
y_test.head(5).tolist()          # -> [0, 0, 1, 0, 1]
```

4/5 match, as expected on a 5-row sample.

---

## Acceptance checklist

- [x] Missing % reported for every affected column, with threshold-rule citation
- [x] `titanic.csv` committed inside `analytics/` via `df.to_csv(..., index=False)`
- [x] Raw dataset loaded **once** (Notebook 1); Notebook 2 uses `pd.read_csv`
- [x] IQR outlier counts for age and fare; fare skewness via mean/median/mode
- [x] 3 bivariate breakdowns + correlation on exactly 6 specified columns
- [x] `adult_male` and `alone` excluded; top-2 abs correlations named
- [x] ≥ 4 multivariate charts, each with interpretation; z-score before/after shown
- [x] Stratified split before any preprocessing, with justification
- [x] Preprocessing fit only on training data (ColumnTransformer inside Pipeline)
- [x] 3 classifiers on identical split; plot_tree labeled; full metric suite
- [x] Baseline vs balanced vs SMOTE comparison with conclusion; SMOTE train-only
- [x] GridSearchCV best params + OOB (RandomForestClassifier(oob_score=True))
- [x] Regression metrics (MAE, RMSE, R², Adj R²) + heteroscedasticity conclusion
- [x] Model comparison table with classification vs regression as separate groups
- [x] Full fitted pipeline saved via `joblib.dump`, reloadable end-to-end