import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


data = pd.read_csv('customer_data.csv')

print(data.head())

data.fillna(data.mean(), inplace=True)

X = data.drop(['YOGESH_CUST_ID', 'YOGESH_CLV'], axis=1)
y = data['YOGESH_CLV']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

print("-------------*****-------------")
print(f'Yogesh Sahu Mean Squared Error:--- {mse}')
