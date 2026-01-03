import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


df = pd.read_csv("C:/Users/dipali/Desktop/ratings.csv")

df['Name'] = pd.factorize(df['Name'])[0]

X = df['Name'].values.reshape(-1, 1)
y = df['Label'].values.reshape(-1, 1)


degree = 3
poly = PolynomialFeatures(degree)
X_poly = poly.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


coeffs = model.coef_.flatten()
intercept = model.intercept_[0]
print(f"f(x) = {coeffs[3]:.4f}x³ + {coeffs[2]:.4f}x² + {coeffs[1]:.4f}x + {intercept:.4f}")


mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")


plt.figure(figsize=(9, 6))


plt.scatter(y_test, y_test, color='green', marker='x', label='Actual', zorder=3)


plt.scatter(y_test, y_pred, color='blue', label='Predicted', zorder=2)


for actual, pred in zip(y_test, y_pred):
    plt.plot([actual, actual], [actual, pred], color='gray', linewidth=0.6, zorder=1)


plt.plot([0, 1], [0, 1], 'r--', label='Perfect Prediction (y = x)', zorder=0)

plt.xlabel('Actual Rating')
plt.ylabel('Predicted Rating')
plt.title('Actual vs Predicted Ratings (Cubic Regression)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure(figsize=(9, 6))
plt.scatter(X, y, color='blue', label='Original Data')


x_line = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
x_line_poly = poly.transform(x_line)
y_curve = model.predict(x_line_poly)

plt.plot(x_line, y_curve, color='red', label='Cubic Regression Curve')
plt.title('Cubic Regression Curve on All Image Ratings')
plt.xlabel('Image Index')
plt.ylabel('Rating Label')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()