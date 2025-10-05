import numpy as np
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

plt.figure(figsize=(7,6))
plt.plot(recall, precision, label=f'PR curve (AP={ap:.3f})')
plt.xlabel('Recall (planet detection rate)')
plt.ylabel('Precision (when we predict planet, how often correct)')
plt.title('Precision–Recall curve (planet as positive)')
plt.grid(True)
plt.legend()
plt.show()

# ---------------------------
# 2) Find threshold that minimizes FNs subject to precision constraint
# ---------------------------
# Define a precision floor you'd like to maintain:
precision_floor = 0.80   # example: maintain at least 80% precision
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
y_pred_default = (proba_pos >= 0.5).astype(int)
print("\nBaseline (threshold=0.5) classification report:")
print(classification_report(y_true_pos, y_pred_default, target_names=['not-planet','planet']))
