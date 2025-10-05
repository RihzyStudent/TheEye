import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score

# Load your trained model
import joblib
model = joblib.load('exoplanet_rf_model.pkl')  # Replace with your model path

# Load original dataset to get feature ranges for synthetic data
df = pd.read_csv('data.csv')

# Features to use for model (excluding Output and Index)
features = ['Orbital Period', 'Transition Duration', 'Transition Depth', 'Planet Rad',
            'Planet Eqbm Temp', 'Stellar Effective Temp', 'Stellar log g', 'Stellar Rad',
            'ra', 'dec']

# Optional: convert Transition Depth to ppm if needed
df['Transition Depth'] = df['Transition Depth'] * 1e-6  # if your model expects ppm as decimal

# Generate synthetic test data
n_samples = 50  # number of synthetic samples
synthetic_data = pd.DataFrame()

for col in features:
    # Sample within the min-max range of each column
    min_val = df[col].min()
    max_val = df[col].max()
    synthetic_data[col] = np.random.uniform(min_val, max_val, n_samples)

# Optional: create synthetic Output for evaluation (if known)
# Here we just randomly assign 0 or 1
synthetic_labels = np.random.randint(0, 2, n_samples)

# Predict using your model
predictions = model.predict(synthetic_data)

# Evaluate
print("Accuracy:", accuracy_score(synthetic_labels, predictions))
print("Classification Report:")
print(classification_report(synthetic_labels, predictions))

# Optionally, include probabilities
probabilities = model.predict_proba(synthetic_data)
print(probabilities[:5])
