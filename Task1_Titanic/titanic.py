import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv("Titanic-Dataset.csv")

print("First 5 rows:")
print(data.head())

data = data[['Survived','Pclass','Sex','Age','Fare']]

data['Sex'] = data['Sex'].map({'male':0,'female':1})

data = data.dropna()

print("\nCleaned Data:")
print(data.head())

sns.countplot(x='Survived', data=data)
plt.title("Titanic Survival Count")
plt.show()

X = data[['Pclass','Sex','Age','Fare']]
y = data['Survived']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = LogisticRegression()
model.fit(X_train,y_train)

predictions = model.predict(X_test)

print("\nPredictions:")
print(predictions)

accuracy = accuracy_score(y_test,predictions)
print("\nModel Accuracy:",accuracy)
