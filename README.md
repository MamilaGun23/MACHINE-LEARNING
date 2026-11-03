# Machine learning coursework

Start with **[Full ML PIPELINE.ipynb](Full%20ML%20PIPELINE.ipynb)** for the verified
end-to-end scikit-learn example: simulated missingness, preprocessing, supervised
L1 selection, training-only hyperparameter search and held-out evaluation.

## Notebook index

| Notebook | Purpose and algorithms | Data | Validation status |
|---|---|---|---|
| [Full ML PIPELINE.ipynb](Full%20ML%20PIPELINE.ipynb) | Mean imputation → scaling → one-vs-rest L1 selection → KNN; GridSearchCV, baseline comparison, macro F1 and confusion matrix | Packaged UCI Iris, 10 deliberately missing sepal-length entries | Fresh execution and fold-boundary assertions passed |
| [LAB 1.ipynb](LAB%201.ipynb) | Data exploration, logistic regression, KNN and K-means | Packaged Iris | Legacy coursework; not rerun in the pinned environment |
| [LAB 2.2.ipynb](LAB%202.2.ipynb) | From-scratch perceptron, learning-rate/epoch experiments and decision regions | Tiny synthetic book examples and binary Iris | Legacy coursework; not rerun |
| [LAB 3 ASSIGNMENT.ipynb](LAB%203%20ASSIGNMENT.ipynb) | Classifier/decision-region comparisons: perceptron, logistic regression, SVM, tree/forest and KNN | Iris and synthetic moons | Legacy coursework; older preprocessing/model-selection patterns need a separate audit |
| [lb4.ipynb](lb4.ipynb) | Missing-data handling, categorical encoding, Wine scaling, L1, sequential backward selection and forest importance | Toy tables and UCI Wine downloads | Legacy coursework; network and modern API compatibility not validated |
| [Practice model 93%.ipynb](Practice%20model%2093%25.ipynb) | Small Keras dense neural network | Iris | Requires separate TensorFlow dependencies; not rerun; filename is not a verified accuracy claim |

Only the primary pipeline is claimed reproducible under the requirements below.
The index describes inspected source; it does not certify every historical output.

## Quick start (Python 3.12)

```sh
git clone https://github.com/MamilaGun23/MACHINE-LEARNING.git
cd MACHINE-LEARNING
python -m venv .venv
```

Activate with `.venv/Scripts/Activate.ps1` on PowerShell or
`source .venv/bin/activate` on macOS/Linux, then:

```sh
python -m pip install -r requirements.txt
python -m notebook "Full ML PIPELINE.ipynb"
```

For a fresh run of all primary-notebook cells:

```sh
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 "Full ML PIPELINE.ipynb"
```

This refreshes real outputs and [results.json](results.json). No credentials,
remote downloads or local dataset paths are needed for the primary notebook.
TensorFlow is intentionally outside this tested environment.

## Data and fitting boundaries

Iris is loaded using `sklearn.datasets.load_iris` (150 flowers, four centimetre
measurements, three species). See its bundled DESCR for original UCI attribution.
The four features are sepal length, sepal width, petal length and petal width.
Rows 0–9 have sepal length replaced by NaN: ten **simulated**, structured missing
entries concentrated in setosa, not real-world missingness or a missing-at-random
experiment.

A stratified 80/20 split (seed 42) reserves 30 test rows. Five shuffled stratified
training folds (seed 42) refit imputation, scaling, L1 selection and KNN inside a
Pipeline. Assertions check actual fold imputer means, scaler sample counts and
the final 120-row refit. The former final `grid_search.fit(X, y)` included held-out
rows; it now fits `X_train, y_train` only.

Explicit one-vs-rest liblinear fits preserve the original multiclass L1 strategy.
Current scikit-learn uses `l1_ratio=1`; a coefficient getter stacks the three
binary estimators' coefficients for SelectFromModel.

The original C candidates (0.01, 0.1, 1, 10) are checked for feasibility using
training folds only. C=0.01 selected zero features in every fold and was excluded.
Remaining C values and k=3,5,7 are searched with errors raised. The selected C=0.1,
k=3 retained sepal width, petal length and petal width. Search CV scores are model
selection results, not unbiased estimates after tuning.

## Executed results (seed 42; 30 held-out rows)

| Model | Training CV accuracy mean ± std | Test accuracy | Test balanced accuracy | Test macro F1 |
|---|---:|---:|---:|---:|
| Most-frequent-class baseline | 0.3333 ± 0.0000 | 0.3333 | 0.3333 | 0.1667 |
| Imputed/scaled KNN, k=3, no selection | 0.9667 ± 0.0167 | 0.9333 | 0.9333 | 0.9327 |
| Tuned L1 + KNN (selection CV) | 0.9667 ± 0.0167 | 0.9333 | 0.9333 | 0.9333 |

The tuned model tied the plain-KNN baseline on test accuracy. Classwise metrics
and confusion matrices are saved in results.json and shown in the notebook.
This small simulated exercise does not establish business impact or broad
generalisation. CV standard deviations are descriptive, not confidence intervals.
No deployment, production readiness or CI result is claimed.

See [REMEDIATION.md](REMEDIATION.md) for the actual checks, changed files and
remaining legacy-notebook limitations.
