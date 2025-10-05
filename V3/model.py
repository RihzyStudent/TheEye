# -----------------------------
# Random Forest Classifier for Exoplanet Detection
# -----------------------------
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
# -----------------------------
# 1. Load dataset
# -----------------------------
data = pd.read_csv('filtered_file.csv')

X = data[['Orbital Period', 'Transit Midpoint', 'Transit Duration', 'Transit Depth', 'Planet Radius','Eqbm Temp', 'Insolation', 'Stellar Temp', 'Stellar Grav', 'Ra', 'Dec']]
y = data['Result']

df_replaced = data.replace(-1, np.nan)

# Compute mean and std per column (NaNs are ignored)
means = df_replaced.mean()
stds = df_replaced.std()

# -----------------------------
# 2. Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, stratify=y
)

# -----------------------------
# 3. Handle class imbalance with SMOTE
# -----------------------------
smote = SMOTE()
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# adasyn = ADASYN()
# X_train_res, y_train_res = adasyn.fit_resample(X_train, y_train)
# -----------------------------
# 4. Train Random Forest with known hyperparameters
# -----------------------------
feature_mean = X_train_res.mean()
feature_std = X_train_res.std()

best_rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    # class_weight="balanced",
    # random_state=42,
    n_jobs=-1
) 
best_rf.fit(X_train_res, y_train_res)

# -----------------------------
# 5. Predict probabilities on test set
# -----------------------------
proba_pos = best_rf.predict_proba(X_test)[:, 1]  # probability of planet
y_true = y_test.values
print("BEST:", best_rf.feature_names_in_)

# -----------------------------
# 6. Threshold tuning
# Maximize recall for class 1 (planet) while keeping recall for class 0 above a floor
# -----------------------------
thresholds = np.linspace(0, 1, 101)  # 0.0, 0.01, ..., 1.0
recall_0_floor = 0.7 # minimum recall for class 0
best_thresh = 0.3
best_metrics = None
best_recall1 = -1

for t in thresholds:
    y_pred = (proba_pos >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0
    recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    if recall_0 >= recall_0_floor:
        if recall_1 > best_recall1:
            best_recall1 = recall_1
            best_thresh = t
            best_metrics = {
                'threshold': t,
                'recall_0': recall_0,
                'recall_1': recall_1,
                'precision_0': tn / (tn + fn) if (tn + fn) > 0 else 0,
                'precision_1': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
            }

# -----------------------------
# 7. Evaluate at selected threshold
# -----------------------------
y_pred_best = (proba_pos >= best_metrics['threshold']).astype(int)

print("Selected threshold and metrics:")
print(best_metrics)

print("\nConfusion matrix:")
print(confusion_matrix(y_true, y_pred_best))

print("\nClassification report:")
print(classification_report(y_true, y_pred_best, target_names=['not-planet','planet']))

# -----------------------------
# 8. Optional: Feature importance
# -----------------------------
import matplotlib.pyplot as plt

importances = best_rf.feature_importances_
feat_names = X.columns

# plt.figure(figsize=(8,6))
# plt.barh(feat_names, importances)
# plt.xlabel("Feature Importance")
# plt.title("Random Forest Feature Importance")
# plt.show()


# def predict_with_outlier_penalty(candidate_df, model, threshold, mean_vals, std_vals, soft_z=2, hard_z=3):
#     # Ensure candidate_df is a DataFrame
#     candidate_df = pd.DataFrame(candidate_df)

    
    
#     # Avoid division by zero in case some std=0
#     safe_std = std_vals.replace(0, 1e-6)
    
#     z_scores = np.abs((candidate_df - mean_vals) / safe_std)
    
#     # Hard rejection for extreme outliers
#     if np.any(z_scores > hard_z):
#         proba = np.array([0.0] * len(candidate_df))
#     else:
#         # Fraction of features outside soft_z
#         punish_ratio = np.clip(np.mean(z_scores > soft_z, axis=1), 0, 1)
#         # Exponential punishment
#         proba = model.predict_proba(candidate_df)[:,1] * np.exp(-punish_ratio * 5)
    
#     pred_class = (proba >= threshold).astype(int)
#     return pred_class, proba

def predict_with_outlier_penalty(candidate_df, model, threshold, mean_vals, std_vals, soft_z=2, hard_z=3):
    """
    Predict exoplanet candidacy with outlier handling.
    - Ignores features with value -1 (treated as missing).
    - Penalizes probabilities if too many features fall outside soft_z range.
    - Hard rejection if any feature exceeds hard_z away from mean.
    """
    candidate_df = pd.DataFrame(candidate_df)

    # Replace std=0 with small epsilon to avoid divide-by-zero
    safe_std = std_vals.replace(0, 1e-6)

    # Mask missing values (-1 means missing)
    mask_missing = (candidate_df == -1)

    # Compute z-scores, ignore -1s by setting them to 0
    z_scores = np.abs((candidate_df - mean_vals) / safe_std)
    z_scores[mask_missing] = 0

    # Hard rejection: if any non-missing value is an extreme outlier
    if np.any((z_scores > hard_z) & ~mask_missing):
        proba = np.array([0.0] * len(candidate_df))
    else:
        # Fraction of *non-missing* features outside soft_z
        punish_ratio = np.clip(
            np.mean((z_scores > soft_z) & ~mask_missing, axis=1), 0, 1
        )
        # Exponential punishment of probability
        raw_proba = model.predict_proba(candidate_df)[:, 1]
        proba = raw_proba * np.exp(-punish_ratio * 5)

    pred_class = (proba >= threshold).astype(int)
    return pred_class, proba



if (__name__ == "__main__"):
    # Example usage
    new_candidate = {
        'Orbital Period': -1,
        'Transit Midpoint': -1,
        'Transit Duration': -1,
        'Transit Depth': -1,
        'Planet Radius': -1,   # instead of Planet Rad
        'Eqbm Temp': -1,      # instead of Planet Eqbm Temp
        'Insolation': -1,
        'Stellar Temp': -1,  # instead of Stellar Effective Temp
        'Stellar Grav': -1,    # instead of Stellar log g
        'Ra': -1,
        'Dec': -1
    }

    new_candidate_df = pd.DataFrame([new_candidate])

    pred_class, prob = predict_with_outlier_penalty(pd.DataFrame([new_candidate]), best_rf, best_metrics['threshold'], feature_mean, feature_std)
    print(f"Predicted: {pred_class}, Probability: {prob}")

    example1 = pd.DataFrame([{
        col: np.random.uniform(X_train_res[col].min(), X_train_res[col].max())
        for col in X_train_res.columns
    }])

    # Example slightly outside range (to test punishment)
    example2 = pd.DataFrame([{
        col: X_train_res[col].max() + 2*X_train_res[col].std() for col in X_train_res.columns
    }])

    # Example completely unrealistic (all zeros)
    example3 = pd.DataFrame([{
        col: 0 for col in X_train_res.columns
    }])


    for i, candidate in enumerate([example1, example2, example3], start=1):
        pred, proba = predict_with_outlier_penalty(
            candidate, best_rf, best_metrics['threshold'], feature_mean, feature_std
        )
        print(f"Example {i}: Predicted={pred[0]}, Probability={proba[0]}")