import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn import metrics
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import joblib
from sklearn.tree import DecisionTreeRegressor

try:
    sales = pd.read_csv('data.csv', encoding="utf-8")
except UnicodeDecodeError:
    sales = pd.read_csv('data.csv', encoding="latin1")
sales.head(40)

sales.dropna(inplace=True)

sales["TotalPrice"] = sales["Quantity"] * sales["UnitPrice"]
X = sales[['Quantity', 'UnitPrice', 'CustomerID']]
y = sales['TotalPrice'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lmodel = DecisionTreeRegressor()
lmodel.fit(X_train, y_train)
joblib.dump(lmodel,"DTmodel.pkl")
y_pred = lmodel.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)



