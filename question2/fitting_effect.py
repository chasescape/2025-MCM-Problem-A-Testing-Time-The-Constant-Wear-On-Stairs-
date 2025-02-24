import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib

matplotlib.use('TkAgg')  # Or use 'Agg' backend

# Your data and reference data
your_data = [3.90, 4.73, 6.05, 2.39, 7.83, 2.54, 5.69, 2.91, 3.64]
reference_data = [3.78, 4.37, 7.65, 2.29, 7.29, 2.21, 6.51, 1.03, 2.72]

# Convert to numpy arrays
your_data = np.array(your_data).reshape(-1, 1)
reference_data = np.array(reference_data)

# Linear regression model
model = LinearRegression()
model.fit(your_data, reference_data)

# Predictions
predictions = model.predict(your_data)

# Calculate Mean Squared Error (MSE) and R-squared (R²)
mse = mean_squared_error(reference_data, predictions)
r2 = r2_score(reference_data, predictions)

# Print evaluation results
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R²): {r2:.4f}")

# Visualize data and fit line
plt.scatter(your_data, reference_data, color='blue', label='Actual Data')
plt.plot(your_data, predictions, color='red', label='Fit Line')
plt.xlabel('Your Data')
plt.ylabel('Reference Data')
plt.legend()
plt.title('Data Fitting')
plt.show()
