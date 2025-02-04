import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns

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
plt.savefig("reports/study_vs_marks.png")
plt.show()
