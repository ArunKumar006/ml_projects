import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv("data.csv")
x=data[["Hours_of_Exercise"]]
y=data["Weight"]

model = LinearRegression()
model.fit(x,y)
prediction = model.predict(x)
print(prediction)