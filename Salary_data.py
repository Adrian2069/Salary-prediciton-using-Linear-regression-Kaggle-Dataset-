import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score 
import matplotlib.pyplot as plt  

Salary = pd.read_csv('Salary_dataset.csv')
print(Salary.info())

# input function and prediction output
X = Salary[['YearsExperience']]
y = Salary ['Salary']

#training and testing sets
X_train, X_test, y_train, y_test = train_test_split (X , y, 
  test_size = 0.4,  #uses 40% of the data for testing
  random_state = 43) #Makes results reproducible 

#creates the Linear Regression model 

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# 6. Evaluate the model
print("\n=== Model Results ===")
print(f"Intercept (β₀): {model.intercept_:.2f}")
print(f"Slope (β₁): {model.coef_[0]:.2f}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}   ← (closer to 1 is better)")

# 7. Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='blue', label='Training data')
plt.scatter(X_test, y_test, color='green', label='Test data')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
plt.title('Salary vs Years of Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Optional: Predict a new value
new_experience = 12
predicted_salary = model.predict([[new_experience]])
print(f"\nPredicted salary for {new_experience} years experience: ${predicted_salary[0]:,.2f}")

                                               
                                    



