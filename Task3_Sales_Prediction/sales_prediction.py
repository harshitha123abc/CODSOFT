import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv("advertising.csv")

print("First 5 rows of dataset:")
print(data.head())

X = data[['TV']]
y = data['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Predicted Sales:", predictions)

plt.scatter(X, y)
plt.plot(X, model.predict(X), linewidth=3)
plt.xlabel("TV Advertising Budget")
plt.ylabel("Sales")
plt.title("Sales Prediction using Linear Regression")
plt.show()
