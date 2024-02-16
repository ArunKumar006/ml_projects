import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


data = pd.read_csv("data.csv")
x=data[["Hours_of_Exercise"]]
y=data["Weight"]
print(x,"lll")
model = LinearRegression()
model.fit(x,y)
predictions = model.predict(x)
print(predictions)
mse = mean_squared_error(y, predictions)
plt.scatter(x, y, color='green')
plt.plot(x, predictions, color='blue', linewidth=2)
plt.xlabel('Hours of Exercise')
plt.ylabel('Weight')
plt.show()
