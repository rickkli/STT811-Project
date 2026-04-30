# STT811-Project

Customer churn modeling project (Telco churn) with an intervention-focused decision engine.

## Notebooks
- `classification_models.ipynb`: trains and evaluates multiple churn classifiers.
- `decision_engine_SHAP.ipynb`: builds a rule + model-based decision engine and explains predictions with SHAP.

## Data
- Primary dataset used in the notebooks: `data/Telco_churn_cleaned.csv`
- Target: `Churn` (1 = churned, 0 = stayed)

## Models Used
The modeling notebook compares several approaches on the same cleaned feature set.

### Classification Models (Core)
- `DummyClassifier(strategy="most_frequent")` as a baseline ("Naive").
- `LogisticRegression(max_iter=1000)` trained on `StandardScaler`-scaled features.
- `RandomForestClassifier(n_estimators=100)` trained on unscaled features.
- `XGBClassifier(eval_metric="logloss")` trained on unscaled features.

### Training / Evaluation Pattern
- Class imbalance handling: `RandomUnderSampler(random_state=42)` before splitting.
- Train/test split: 80/20 (`train_test_split(..., test_size=0.2, random_state=42)`).
- Metrics/plots used in the notebook include:
  - Classification reports and confusion matrices (per model).
  - ROC curves (AUC via `roc_auc_score`) to compare models on probability outputs.
  - Precision-recall curve (shown for logistic regression) and PR-AUC.
  - Calibration curve (reliability diagram) for probability calibration sanity checks.
- Hyperparameter tuning: `GridSearchCV` is used to search parameter grids for LR/RF/XGBoost.
- Visualization: PCA to 2D (`PCA(n_components=2)`) for decision boundary plots (interpretability/teaching aid, not the production feature space).

### Survival Analysis (Exploratory)
- Cox proportional hazards model via `lifelines.CoxPHFitter(penalizer=0.1)`, fit as:
  - `duration_col="Tenure Months"` (time),
  - `event_col="Churn"` (event indicator),
  - with one-hot encoding via `pd.get_dummies(..., drop_first=True)`.

## Decision Engine (How It Works)
Implemented in `decision_engine_SHAP.ipynb`. The decision engine is designed to turn a churn probability into an action recommendation (or no action).

### 1) Risk Screening (Model)
- Uses the trained/scaled logistic regression model (`lr_model` + `scaler`) to compute:
  - `churn_prob = lr_model.predict_proba(scaled_row)[0][1]`
- At-risk threshold: `THRESHOLD = 0.5`
  - If `churn_prob < 0.5`, the engine returns "no action needed".

### 2) Expected Loss Estimate (Business Proxy)
For at-risk customers, the engine estimates potential loss using tenure and monthly charges:
- Computes a demographic-group average tenure from *non-churn* customers using:
  - `Gender`, `Senior Citizen`, `Partner`, `Dependents`
  - Groups with fewer than `MIN_GROUP_COUNT = 20` fall back to the overall non-churn average.
- Remaining tenure:
  - If current tenure exceeds the group average, it caps remaining tenure at `min(12, 72 - tenure)`.
  - Otherwise, `remaining = (group_avg_tenure - tenure)`.
- Loss calculation:
  - `expected_loss = remaining * Monthly Charges * MARGIN_RATE`, where `MARGIN_RATE = 0.2`
  - `expected_loss_adj = expected_loss * TENURE_ASSUMPTION`, where `TENURE_ASSUMPTION = 0.5`
- The engine also computes a benchmark `loss_median` over the at-risk population (`churn_prob > 0.5`) and labels the customer as:
  - `Upper` if `expected_loss >= loss_median`, else `Lower`.

### 3) Response Grade (A-D)
The grade combines probability level and expected loss level:
- Probability:
  - `High` if `churn_prob >= 0.7`
  - `Mid` if `0.5 <= churn_prob < 0.7`
- Loss: `Upper` vs `Lower` (relative to `loss_median`)

Grade mapping (as coded):
- `A` = High + Upper: recommend a top-2 action combo + active financial support.
- `B` = High + Lower: recommend a top-1 action + minor financial support.
- `C` = Mid + Upper: recommend a top-1 action + minor financial support.
- `D` = Mid + Lower: email notification only (no incentives).

### 4) Action Simulation (Counterfactual Recommendations)
For grades A/B/C, the engine searches for actions that reduce churn risk by simulating feature changes and re-scoring with the model.

Actions are defined as edits to one-hot feature columns, grouped to avoid recommending two changes from the same category:
- Add-on services:
  - Online Security, Online Backup, Device Protection, Tech Support
- Contract:
  - Switch to 1-year, switch to 2-year
- Payment method:
  - Switch away from electronic check to mailed check / bank transfer / credit card

Simulation logic:
- For each action, apply the one-hot edits, re-scale, and compute `Prob After`.
- Score actions by `Reduction = churn_prob - Prob After` and sort descending.
- If the best single action achieves `Prob After < 0.5`, the engine prints the top 5 single actions.
- Otherwise, it also simulates a "Top-2 combo" consisting of:
  - the best action overall (Top 1), plus
  - the best action from a different action group (Top 2),
  - and prints the combo alongside the top single actions.

Batch mode:
- `run_batch(n, seed=42)` samples at-risk customers and prints a summary table including grade, incentive, top action, and predicted reduction.

## Explainability (SHAP)
`decision_engine_SHAP.ipynb` uses SHAP to explain the logistic regression score:
- Explainer: `shap.LinearExplainer(lr_model, X_train_scaled_df)` (log-odds / margin space).
- Visuals include:
  - A gauge of predicted churn risk for a selected customer.
  - A bar view of per-feature SHAP contributions (red increases churn risk, green decreases).
  - A waterfall view from baseline log-odds to the customer's final log-odds (then mapped through the sigmoid to probability).

## Webapp Prototype
Prototype web app for interactive scoring and recommendations:
- Live link: https://customerchurn-ar5ygvrsgvdqsxh89bqejn.streamlit.app/