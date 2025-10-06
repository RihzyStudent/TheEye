import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
import numpy.linalg as la
import scipy as sc
warnings.filterwarnings('ignore')

data = pd.read_csv('filtered_file.csv')

# titanic_data = titanic_data.dropna(subset=['Survived'])

X = data[['Orbital Period', 'Transit Midpoint', 'Transit Duration', 'Transit Depth', 'Planet Radius','Eqbm Temp', 'Insolation', 'Stellar Temp', 'Stellar Grav', 'Ra', 'Dec']]
y = data['Result']
# print(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01)

rf_classifier = RandomForestClassifier(n_estimators=100)

# print(X_test)
rf_classifier.fit(X_train, y_train)

y_pred = rf_classifier.predict( X_test)
6
print(y_pred)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_rep)

sample = X_test.iloc[0:1]
prediction = rf_classifier.predict(sample)

print(prediction)
sample_dict = sample.iloc[0].to_dict()
print(f"\nSample Passenger: {sample_dict}")
print(f"Predicted Survival: {'Survived' if prediction[0] == 1 else 'Did Not Survive'}")