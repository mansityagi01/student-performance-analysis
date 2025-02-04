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
sns.boxplot(x=df["number_courses"], y=df["Marks"], palette="coolwarm")
plt.title("Number of Courses vs Marks")
plt.xlabel("Number of Courses")
plt.ylabel("Marks")
plt.savefig("Course_VS_Marks.png")




plt.show()


