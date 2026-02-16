

# import flask
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score
import joblib

data = pd.read_csv('data.csv')
data = data.dropna()

x = data[['lat', 'lon', 'magX', 'magY', 'magZ', 'mag_val', 'floor']]
y = data[['mageX', 'imageY']]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train a multi-output regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")



joblib.dump(model, '../models/linear_model2.pkl')

#linear_model2 -> floor data was used in training