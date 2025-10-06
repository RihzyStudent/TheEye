import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

california_housing = fetch_california_housing()
california_data = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
california_data['MEDV'] = california_housing.target


X = california_data.drop('MEDV', axis=1)
y = california_data['MEDV']



titanic_data = pd.read_csv('data.csv')

# titanic_data = titanic_data.dropna(subset=['Survived'])

X = titanic_data[['Orbital Period', 'Transition Duration', 'Transition Depth', 'Planet Rad', 'Planet Eqbm Temp', 'Stellar Effective Temp', 'Stellar log g', 'Stellar Rad', 'ra', 'dec']]
y = titanic_data['Output']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print()


rf_regressor = RandomForestRegressor(n_estimators=10, random_state=42)

rf_regressor.fit(X_train, y_train)

y_pred = rf_regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

single_data = X_test.iloc[0].values.reshape(1, -1)
predicted_value = rf_regressor.predict(single_data)
print(f"Predicted Value: {predicted_value[0]:.2f}")
print(f"Actual Value: {y_test.iloc[0]:.2f}")

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")