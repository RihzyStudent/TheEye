# Random Forest Classifier with Imbalance Handling
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# -----------------------------
# 1. Load your dataset
# Replace with your actual dataset
# Example:
# X = df.drop("target", axis=1)
# y = df["target"]
# -----------------------------

titanic_data = pd.read_csv('data.csv')

# titanic_data = titanic_data.dropna(subset=['Survived'])

X = titanic_data[['Orbital Period', 'Transition Duration', 'Transition Depth', 'Planet Rad', 'Planet Eqbm Temp', 'Stellar Effective Temp', 'Stellar log g', 'Stellar Rad', 'ra', 'dec']]
y = titanic_data['Output']

# For demonstration, I'll assume you already have X, y defined.

# -----------------------------
# 2. Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# -----------------------------
# 3. Handle imbalance (optional, try with/without SMOTE)
# -----------------------------
smote = SMOTE( random_state= 42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# -----------------------------
# 4. Define Random Forest with class weighting
# -----------------------------
rf = RandomForestClassifier( random_state=42, n_jobs=-1, min_samples_split=2, min_samples_leaf=1)

# -----------------------------
# 5. Hyperparameter tuning with GridSearchCV
# -----------------------------
param_grid = {
    "n_estimators": [200],
    "max_depth": [None],
    "min_samples_split": [2],
    "min_samples_leaf": [1]
}

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring="recall_macro",   # can use 'recall_macro' if you care more about recall
    cv=10,
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train_res, y_train_res)

print("Best Parameters:", grid_search.best_params_)

# -----------------------------
# 6. Evaluate on test set
# -----------------------------
best_rf = grid_search.best_estimator_

y_pred = best_rf.predict(X_test)

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -----------------------------
# 7. Optional: Adjust threshold
# -----------------------------
y_proba = best_rf.predict_proba(X_test)[:, 1]
threshold = 0.4  # lower threshold to improve recall for class 0
y_pred_thresh = (y_proba > threshold).astype(int)

print("\nClassification Report (Adjusted Threshold = 0.4):\n",
      classification_report(y_test, y_pred_thresh))


# import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, classification_report, confusion_matrix

# Assumptions:
# - best_rf is your trained classifier (RandomForest or other)
# - X_test, y_test are your test set
# - planet label is 1 (positive). If planet==0, change idx=0 below.

pos_label = 1
proba_pos = best_rf.predict_proba(X_test)[:, 1]   # prob of class 1 (planet)
y_true_pos = (y_test == pos_label).astype(int)    # binary: 1 if planet, else 0

# ---------------------------
# 1) Precision-Recall curve
# ---------------------------
precision, recall, thresholds = precision_recall_curve(y_true_pos, proba_pos)
ap = average_precision_score(y_true_pos, proba_pos)

# plt.figure(figsize=(7,6))
# plt.plot(recall, precision, label=f'PR curve (AP={ap:.3f})')
# plt.xlabel('Recall (planet detection rate)')
# plt.ylabel('Precision (when we predict planet, how often correct)')
# plt.title('Precision-Recall curve (planet as positive)')
# plt.grid(True)
# plt.legend()
# plt.show()

# ---------------------------
# 2) Find threshold that minimizes FNs subject to precision constraint
# ---------------------------
# Define a precision floor you'd like to maintain:
precision_floor = 0.80  # example: maintain at least 80% precision
# You can change precision_floor to reflect how many false alarms you tolerate.

best_thresh = 0.5
best_fn = np.inf
best_metrics = None

# thresholds returned by precision_recall_curve are cutpoints for prob > t => positive.
# Note: thresholds length = len(precision)-1
for i, t in enumerate(thresholds):
    # predict positive if prob >= threshold t
    y_pred = (proba_pos >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_pos, y_pred).ravel()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # only consider thresholds that keep precision >= floor
    if prec >= precision_floor:
        # primary objective: minimize FN (so minimize fn)
        if fn < best_fn:
            best_fn = fn
            best_thresh = t
            best_metrics = dict(threshold=t, precision=prec, recall=rec, fn=fn, fp=fp, tp=tp, tn=tn)

# If no threshold met the precision floor, relax requirement and choose best recall point (lowest FN)
if best_metrics is None:
    print(f"No threshold satisfied precision >= {precision_floor:.2f}. Relaxing precision constraint.")
    # choose threshold giving highest recall (lowest FN)
    best_idx = np.argmax(recall)
    best_thresh = thresholds[best_idx-1] if best_idx>0 else 0.0
    y_pred = (proba_pos >= best_thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_pos, y_pred).ravel()
    best_metrics = dict(threshold=best_thresh, precision=tp/(tp+fp) if (tp+fp)>0 else 0.0,
                        recall=tp/(tp+fn) if (tp+fn)>0 else 0.0, fn=fn, fp=fp, tp=tp, tn=tn)

# ---------------------------
# 3) Results
# ---------------------------
print("Selected threshold and metrics (minimize FN subject to precision constraint):")
print(best_metrics)

# Full classification report using selected threshold
y_pred_best = (proba_pos >= best_metrics['threshold']).astype(int)
print("\nConfusion matrix (planet positive):")
print(confusion_matrix(y_true_pos, y_pred_best))
print("\nClassification report (planet as positive):")
print(classification_report(y_true_pos, y_pred_best, target_names=['not-planet','planet']))

# Also show default 0.5 baseline for comparison
y_pred_default = (proba_pos >= 0.2).astype(int)
print("\nBaseline (threshold=0.2) classification report:")
print(classification_report(y_true_pos, y_pred_default, target_names=['not-planet','planet']))
