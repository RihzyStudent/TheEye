import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

titanic_data = pd.read_csv('data.csv')

# titanic_data = titanic_data.dropna(subset=['Survived'])

X = titanic_data[['Orbital Period', 'Transition Duration', 'Transition Depth', 'Planet Rad', 'Planet Eqbm Temp', 'Stellar Effective Temp', 'Stellar log g', 'Stellar Rad', 'ra', 'dec']]
y = titanic_data['Output']

# X.loc[:, 'Sex'] = X['Sex'].map({'female': 0, 'male': 1})

# X.loc[:, 'Age'].fillna(X['Age'].median(), inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

rf_classifier = RandomForestClassifier(n_estimators=10000)

rf_classifier.fit(X_train, y_train)

y_pred = rf_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_rep)

# sample = X_test.iloc[0:1]
# prediction = rf_classifier.predict(sample)

# sample_dict = sample.iloc[0].to_dict()
# print(f"\nSample Passenger: {sample_dict}")
# print(f"Predicted Survival: {'Survived' if prediction[0] == 1 else 'Did Not Survive'}")