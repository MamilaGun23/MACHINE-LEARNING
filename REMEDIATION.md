# Remediation evidence — 3 November 2026

Changed Full ML PIPELINE.ipynb and README.md; added requirements.txt,
.python-version, .gitignore and measured results.json. The README now indexes
all six existing notebooks with purpose, data, algorithms and validation status.

## Verified corrections

The existing sklearn Pipeline already protected imputation, scaling and L1
selection within each CV fit. However, the final original grid search called
grid_search.fit(X, y), including rows used in the earlier held-out test. It now
fits X_train/y_train only, using a single explicit stratified 120/30 split
(seed 42) and five shuffled training folds (seed 42).

L1 liblinear selection remains one-vs-rest via an explicit wrapper and coefficient
getter. l1_ratio=1 avoids deprecated penalty arguments. Training-only feasibility
checks found that C=0.01 selected no features in all five folds; that candidate is
excluded before searching the remaining original C values and k=3,5,7.

Actual fold estimators are audited: imputer means match fold training data,
scaler means/sample counts match imputed fold training rows, and the final refit
has exactly 120 training samples. The ten missing entries are explicitly
simulated in rows 0–9 of sepal length, concentrated in setosa.

Fixed majority-class and plain-KNN baselines share the training folds and final
test rows. Final reporting includes accuracy, balanced accuracy, macro F1,
classwise metrics and confusion matrices. Model-selection CV is labelled
separately from held-out results.

## Actual validation

CPython 3.12.16, scikit-learn 1.9.1, NumPy 2.5.4.

- `python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 "Full ML PIPELINE.ipynb"`:
  all cells executed and all fitting-boundary/feasibility assertions passed.
- Four code cells compiled; saved outputs contain no execution errors.
- `git diff --check`: passed.

The selected C=0.1/k=3 model and plain KNN both scored 0.9333 test accuracy on 30
rows. Their macro F1 values were 0.9333 and 0.9327 respectively; the majority-class
baseline had 0.3333 accuracy. README metrics come from the executed results.json.
No pytest package was added: boundary assertions live beside the actual pipeline
in the executable notebook.

Windows blocked the venv launcher, so checks used a separate newly provisioned
Python interpreter. Standard venv activation was not fully exercised on this host.

## Remaining legacy issues and limits

Other notebooks were inspected for the README index, but were not repaired or
executed under these requirements:

- LAB 3 ASSIGNMENT.ipynb:1200 and :1207 pass pre-scaled arrays into CV grid
  searches. A separate audit should move scaling inside those CV pipelines.
- lb4.ipynb:631 uses multiclass liblinear L1 directly; compatibility with the
  pinned modern sklearn environment is not established.
- Practice model 93%.ipynb:28 and :53 requires TensorFlow (not in the tested
  environment) and does not seed neural-network training. Its filename is not
  evidence of a measured 93% result.
- LAB 1.ipynb and LAB 2.2.ipynb remain historical exercises, not rerun here.

Safe to describe the primary notebook as a reproducible sklearn preprocessing/
selection/classification pipeline with training-only search, baseline comparison
and verified fitting boundaries. Do not describe simulated missingness as real
operational data, all legacy notebooks as validated, a fixed test score as a
general guarantee, or the repository as deployed/production-ready/CI-verified.
