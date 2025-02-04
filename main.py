import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

df = pd.read_csv("Student_Marks.csv")    #Load dataset

# Display first few rows
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Summary statistics
print(df.describe())

# Calculate mean and standard deviation
mean_marks = df["Marks"].mean()
std_marks = df["Marks"].std()

# Define outlier threshold (3 standard deviations away from the mean)
lower_bound = mean_marks - (3 * std_marks)
upper_bound = mean_marks + (3 * std_marks)

# Detect outliers
outliers = df[(df["Marks"] < lower_bound) | (df["Marks"] > upper_bound)]
print(outliers)

#for detecting outliers
plt.figure(figsize=(10, 5))
sns.scatterplot(x=df["time_study"], y=df["Marks"], color="red")
plt.title("Study Time vs Marks")
plt.xlabel("Time Studied (hours)")
plt.ylabel("Marks")
plt.savefig("scatter_plot.png") 




plt.figure(figsize=(14, 5))

# Histogram
plt.subplot(1, 2, 1)
sns.histplot(df["Marks"], bins=20, kde=True, color="blue")
plt.title("Distribution of Marks")
plt.xlabel("Marks")
plt.ylabel("Frequency")

# Boxplot
plt.subplot(1, 2, 2)
sns.boxplot(y=df["Marks"], color="orange")
plt.title("Boxplot of Marks")
plt.savefig("marks_distribution.png") 

#Relationship Between Study Time and Marks
plt.figure(figsize=(10, 5))
sns.regplot(x=df["time_study"], y=df["Marks"], color="green")
plt.title("Study Time vs Marks (with Regression Line)")
plt.xlabel("Time Studied (hours)")
plt.ylabel("Marks")
plt.savefig("Regression line.png")

#relationship between number of courses and marks 
plt.figure(figsize=(10, 5))
sns.boxplot(x=df["number_courses"], y=df["Marks"], hue=df["number_courses"], palette="coolwarm", legend=False)

plt.title("Number of Courses vs Marks")
plt.xlabel("Number of Courses")
plt.ylabel("Marks")
plt.savefig("Course_VS_Marks.png")


# Correlation Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=2)
plt.title("Correlation Matrix")
plt.savefig("correlation_matrix.png")


# Creating a new feature: Marks per Course
df["marks_per_course"] = df["Marks"] / df["number_courses"]

# Display first few rows with the new column
print(df.head())

# Save the updated dataset
df.to_csv("Updated_Student_Marks.csv", index=False)

# Define features and target variable
X = df[["time_study", "number_courses", "marks_per_course"]]
y = df["Marks"]


# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate model
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Model Performance:")
print(f"R² Score: {r2:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")


# Scatter plot: Actual vs Predicted Marks
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color="blue", label="Predictions")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle="dashed", color="red", label="Perfect Fit") 
plt.xlabel("Actual Marks")
plt.ylabel("Predicted Marks")
plt.title("Actual vs Predicted Marks")
plt.legend()
plt.savefig("Actual_vs_Predicted.png") 

# Extract feature importance from the trained model
feature_importance = np.abs(model.coef_)  # Get absolute values of coefficients
features = X_train.columns

# Create a DataFrame for better visualization
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importance   for more accurate predictions
plt.figure(figsize=(8, 5))
sns.barplot(x=importance_df['Importance'], y=importance_df['Feature'], palette="viridis")
plt.title("Feature Importance in Predicting Marks")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.savefig("Feature_Importance.png")

plt.show()
print(importance_df)

# Save the trained model to a file
joblib.dump(model, "student_marks_predictor.pkl")

print("Model saved successfully!")



