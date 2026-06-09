# MD2

Frozen coursework: second Data Mining (MD) project at FIB-UPC. Classification and EDA on a user dataset using KNN, Naive Bayes, SVM, and decision trees with cross-validation and grid-search hyperparameter tuning.

## Architecture

- `src/` — one Jupyter notebook per classifier (`KNN.ipynb`, `naiveBayes.ipynb`, `SVM.ipynb`, `decision_tree.ipynb`, `meta_learning.ipynb`) plus EDA notebooks (`univariate_analisis.ipynb`, `Bivariate.ipynb`, `nullables.ipynb`) and shared `dataLoader.py` (loading + KNN evaluation helpers).
- `data/allUsers.lcl.csv` — input dataset; `docs/` — report and slides; `Makefile` bundles deliverables into `project.tar`.

## Build and Test

- Python 3.11 via Pipenv: `pipenv install` then `pipenv run jupyter notebook` (or `pip install -r requirements.txt`).
- No automated tests; run notebooks in `src/` interactively.
- `make` builds `project.tar`; `make clean` removes it.

## Pitfalls

- Frozen coursework — do not refactor or modernize. Notebooks reference relative paths to `data/`, so run them from `src/`.

See [README.md](README.md).
