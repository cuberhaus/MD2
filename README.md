# MD2

Second project for the Data Mining (Mineria de Dades) course at FIB-UPC. Classification and exploratory data analysis on a user dataset using multiple machine learning algorithms.

## Overview

The project performs end-to-end data analysis: data loading, null handling, univariate/bivariate analysis, and classification using several algorithms with cross-validation and grid search for hyperparameter tuning.

## Structure

```
├── src/
│   ├── dataLoader.py              # Data loading and KNN evaluation utilities
│   ├── KNN.ipynb                  # K-Nearest Neighbors classifier
│   ├── naiveBayes.ipynb           # Naive Bayes classifier
│   ├── SVM.ipynb                  # Support Vector Machine classifier
│   ├── decision_tree.ipynb        # Decision tree classifier
│   ├── meta_learning.ipynb        # Meta-learning analysis
│   ├── univariate_analisis.ipynb  # Univariate EDA
│   ├── Bivariate.ipynb            # Bivariate EDA
│   └── nullables.ipynb            # Null value handling
├── data/                          # Dataset (allUsers.lcl.csv)
├── MD.mp4                         # Presentation video
├── Makefile                       # Builds project.tar with docs and code
├── Pipfile                        # Python 3.11 dependencies
└── Pipfile.lock
```

## Tech Stack

- **Python 3.11** with scikit-learn, pandas, NumPy, matplotlib
- **Jupyter Notebooks** for analysis
- **Pipenv** for dependency management
