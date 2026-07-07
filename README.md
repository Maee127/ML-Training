# ML-Training — Classification Studies

A focused exploration of classification models and hyperparameter tuning using the scikit-learn breast cancer dataset. 
This study compares classical ML models and gradient-boosting learners, documents training logs (CatBoost), and includes visual comparisons of model performance. 
*The project is intended for researchers and practitioners who want a repeatable notebook-driven study of classification workflows, feature processing, model comparison, and simple tuning strategies.*

## Highlights
- Goal: improve classification score (accuracy / F1) — target noted in the notebook as > 0.99.
- Dataset: sklearn.datasets.load_breast_cancer (built-in).
- Models explored: Logistic Regression, SVM, GaussianNB, Decision Tree, Random Forest, K-Nearest Neighbors, XGBoost, LightGBM, CatBoost (and others).
- Key techniques: train/test split, scaling, feature selection (SelectKBest, RFE), PCA, model comparison, metrics (accuracy, precision, recall, F1, confusion matrix), and CatBoost training logs captured in catboost_info.
- Artifacts: Jupyter notebook `Classification.ipynb`, CatBoost logs under `catboost_info/`, and a model comparison graphic `model_comparison_rotated.png`.

---

## Repository structure (top-level)
```
Classification.ipynb            # Main notebook: data prep, model training, evaluation, plots
README.md                      # (this file)
catboost_info/                 # CatBoost training logs and metrics (json, .tsv, etc.)
  ├─ catboost_training.json    # training iterations & metrics
  ├─ learn_error.tsv
  └─ time_left.tsv
model_comparison_rotated.png   # Visualization comparing model metrics
requirements.txt               # project reuirements packages
```

---

## Notebook summary (Classification.ipynb)
The notebook walks through a typical classification experiment pipeline:
1. Imports and environment setup (numpy, pandas, matplotlib, seaborn, scipy).
2. Load dataset (breast cancer) and create DataFrame.
3. Exploratory data analysis and summary statistics.
4. Preprocessing: scaling (StandardScaler), optional transformations (Yeo-Johnson), feature selection (SelectKBest, mutual_info_classif, RFE), and PCA experiments.
5. Model training and evaluation of many learners:
   - LogisticRegression, SVC, GaussianNB
   - DecisionTreeClassifier, RandomForestClassifier
   - KNeighborsClassifier
   - Boosting libraries: XGBoost (XGBClassifier), LightGBM (LGBMClassifier), CatBoost (CatBoostClassifier)
6. Metrics: accuracy_score, f1_score, precision, recall, confusion matrix, classification_report.
7. Model comparison plot(s) exported (one provided as model_comparison_rotated.png).
8. CatBoost logs captured to `catboost_info/` showing iteration-level learn metrics (e.g., Logloss).

---

## How to run locally

Prerequisites:
- Python 3.10+ (the notebook was run with a modern Python — ensure a recent environment)
- Jupyter (Lab or Notebook) or VS Code with Jupyter support

Suggested quick-start commands:
```bash
# Clone the repo
git clone https://github.com/Maee127/ML-Training.git
cd ML-Training

# (Optional) Create and activate virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# Install required packages (example list)
pip install numpy pandas matplotlib seaborn scipy scikit-learn xgboost lightgbm catboost jupyterlab

# Start Jupyter and open the notebook
jupyter lab
# or
jupyter notebook
```

Then open `Classification.ipynb` and run cells top-to-bottom. The notebook is self-contained: it uses the sklearn breast cancer dataset (no external downloads required).

---

## Reproducing experiments & notes
- Use a fixed random seed (e.g., `random_state=42`) for reproducibility across train/test splits and stochastic learners.
- For gradient-boosting models, training artifacts and iteration metrics for CatBoost are saved under `catboost_info/`. The `catboost_training.json` contains per-iteration Logloss values (learn metric) and timing estimates.
- The repository includes a pre-generated performance comparison image: `model_comparison_rotated.png`. Re-run plotting cells in the notebook to regenerate with your current runs.
- When tuning, use cross-validation (GridSearchCV or RandomizedSearchCV) and monitor validation metrics to avoid overfitting. Consider stratified folds for class balance.

---

## Results (what to expect)
- Training logs (CatBoost) show rapid reduction of Logloss over iterations (see `catboost_info/catboost_training.json`), indicating strong fit on the breast cancer dataset.
- The notebook calculates standard classification metrics (accuracy, F1, precision, recall) and shows model comparisons via plots. Inspect the notebook outputs and `model_comparison_rotated.png` to view summarized comparisons.

---

## Tips to improve model performance (suggested experiments)
- Robust cross-validation: StratifiedKFold with repeats and metric averaging to better estimate generalization.
- Feature engineering: interaction terms, domain-driven transformations, dimensionality reduction (PCA) followed by model ensembling.
- Hyperparameter tuning: RandomizedSearchCV followed by a focused GridSearchCV for top models (CatBoost, XGBoost, LightGBM, RandomForest).
- Calibration: try probability calibration (CalibratedClassifierCV) if probabilities are important.
- Handle class imbalance (if present): though breast cancer dataset is relatively balanced, try stratified sampling, class weights, or resampling techniques as experiments.

---

## Contributing
- Add experiments or notebooks to expand coverage of preprocessing, tuning, or deployment.
- If you add long-running training scripts, include checkpoints and small reproducible debug configurations.
- Open issues/pull requests with reproducible steps and a short summary of findings.

---

## License & Contact
- License: MIT
- Author / Contact: Maedeh Torkian — 
LinkedIn: https://www.linkedin.com/in/maedeh-torkian/
Email: Maede.torkian@gmail.com

---


