
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

X_train = np.array([0,1,2,3,4,5,6,7,8,9]).reshape(-1, 1)
y_train = np.array([0,0.5,1,1.5,2,2.5,3,3.5,4,4.5])

X_test = np.array([10,11,12,13,14]).reshape(-1, 1)
y_test = np.array([5,5.5,7,6.5,7])

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

plt.scatter(X_train, y_train, label="Training Data")
plt.scatter(X_test, y_test, label="Test Data")
plt.plot(X_test, y_pred, label="Regression Line")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Regression Analysis")
plt.legend()
plt.show()

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R² Score:", r2)
