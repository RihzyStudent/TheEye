import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset from CSV
df = pd.read_csv("data.csv")

# Assume the last column is the target, and all others are features
X = df.iloc[:, 1:-1]   # all columns except last
y = df.iloc[:, -1]    # last column as target

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the Random Forest Regressor
rf = RandomForestRegressor(
    n_estimators=100,     # number of trees
    random_state=42
)
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)

# Evaluation
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))
