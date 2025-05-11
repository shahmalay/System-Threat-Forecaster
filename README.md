# 🧠 Machine Learning Notebook – 22f2001484_notebook_t12025

## 📘 Overview

This Jupyter Notebook demonstrates a complete machine learning workflow, including data preprocessing, model training, evaluation, and hyperparameter tuning using **Scikit-learn**.

It likely focuses on classification tasks and explores multiple models including:
- Logistic Regression
- Random Forest
- Gradient Boosting
- Dummy Classifiers (for baseline comparison)

## 📂 File Structure

- `22f2001484_notebook_t12025.ipynb` – Main notebook with all code and outputs.
- `README.md` – This file.
- `requirements.txt` – (Optional) Can be generated based on libraries used.

## 📊 Key Features

- ✅ Data Cleaning and Imputation with `SimpleImputer`
- ✅ Column-wise Transformations using `ColumnTransformer`
- ✅ Baseline Modeling with `DummyClassifier`
- ✅ Model Building with:
  - `LogisticRegression`
  - `RandomForestClassifier`
  - `HistGradientBoostingClassifier`
- ✅ Evaluation using `accuracy_score`
- ✅ Hyperparameter tuning using `GridSearchCV`

## 🧰 Dependencies

Below are the major Python libraries used:

```python
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
