import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Step 1: Read the Excel file
file_path = "C:/Users/caronet/Downloads/baseball.xlsx"
data = pd.read_excel(file_path)

# Step 2: Perform linear regression for Wins vs. Runs Difference
# Dependent variable: Wins (column F)
# Independent variable: Difference between Runs Scored (column D) and Runs Allowed (column E)
X1 = data['Runs Scored'] - data['Runs Allowed']
Y1 = data['Wins']
X1 = X1.values.reshape(-1, 1)  # Reshape to make it compatible with sklearn

# Create and fit the linear regression model
model1 = LinearRegression()
model1.fit(X1, Y1)

# Predictions using the model
Y1_pred = model1.predict(X1)

# Calculate R-squared value
r_squared1 = r2_score(Y1, Y1_pred)

# Plot the data and regression line
plt.figure(figsize=(8, 6))
plt.scatter(X1, Y1, color='blue', label='Data Points')
plt.plot(X1, Y1_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel('Runs Difference (Runs Scored - Runs Allowed)')
plt.ylabel('Wins')
plt.title('Wins vs. Runs Difference')
plt.text(0.5, 0.95, f'R-squared: {r_squared1:.2f}', transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
plt.legend()
plt.show()

# Step 3: Perform linear regression for Runs Difference vs. Team Batting Average
# Dependent variable: Difference between Runs Scored (column D) and Runs Allowed (column E)
# Independent variable: Team Batting Average (column I)
X2 = data['Team Batting Average']
Y2 = data['Runs Scored'] - data['Runs Allowed']
X2 = X2.values.reshape(-1, 1)  # Reshape to make it compatible with sklearn

# Create and fit the linear regression model
model2 = LinearRegression()
model2.fit(X2, Y2)

# Predictions using the model
Y2_pred = model2.predict(X2)

# Calculate R-squared value
r_squared2 = r2_score(Y2, Y2_pred)

# Plot the data and regression line
plt.figure(figsize=(8, 6))
plt.scatter(X2, Y2, color='blue', label='Data Points')
plt.plot(X2, Y2_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel('Team Batting Average')
plt.ylabel('Runs Difference (Runs Scored - Runs Allowed)')
plt.title('Runs Difference vs. Team Batting Average')
plt.text(0.5, 0.95, f'R-squared: {r_squared2:.2f}', transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
plt.legend()
plt.show()

# Step 4: Perform multiple linear regression
# Dependent variable: Difference between Runs Scored (column D) and Runs Allowed (column E)
# Independent variables: OBP (column G) and SLG (column H)
X3 = data[['OBP', 'SLG']]
Y3 = data['Runs Scored'] - data['Runs Allowed']

# Create and fit the linear regression model
model3 = LinearRegression()
model3.fit(X3, Y3)

# Print regression statistics
print("Multiple Regression Statistics:")
print("Intercept:", model3.intercept_)
print("Coefficients:", model3.coef_)
print("R-squared:", model3.score(X3, Y3))
