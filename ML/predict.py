# predict.py
import pandas as pd
import joblib
import numpy as np

# -----------------------------
# Load the saved model package
# -----------------------------
model_package = joblib.load('exoplanet_rf_model.pkl')
model = model_package['model']
threshold = (model_package['threshold'])*0.90
# print(threshold0
feature_mean = model_package['feature_mean']
feature_std = model_package['feature_std'] #standard deviation


def predict(candidate_df:pd.DataFrame, soft_z:float=2.5, hard_z:float=5.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Predicts using the RandomForest model with outlier penalties.
    Soft penalty reduces probability for moderate outliers.
    Hard penalty caps extreme outliers.
    
    Parameters:
        candidate_df (pd.DataFrame): DataFrame with same features as training
        soft_z (float): Z-score threshold for soft punishment
        hard_z (float): Z-score threshold for extreme outliers
        
    Returns:
        pred_class (np.array): 0/1 predictions
        prob (np.array): predicted probabilities
    """
    candidate_df = pd.DataFrame(candidate_df)
    safe_std = feature_std.replace(0, 1e-6)
    z_scores = np.abs((candidate_df - feature_mean) / safe_std)
    
    punish_ratio = np.clip(np.mean(np.minimum(z_scores / soft_z, 1), axis=1), 0, 1)    
    extreme_ratio = np.clip(np.mean(np.minimum(z_scores / hard_z, 1), axis=1), 0, 1)
    
    raw_proba = model.predict_proba(candidate_df)[:, 1]
    
    prob = raw_proba * (1 - 0.5 * punish_ratio) * (1 - 0.5 * extreme_ratio)
    prob = np.clip(prob, 0, 1)
    
    pred_class = (prob >= threshold).astype(int)
    return pred_class, prob

 
