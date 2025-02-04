import joblib
import numpy as np
import pandas as pd  # Import Pandas

# Load the saved model
loaded_model = joblib.load("student_marks_predictor.pkl")

# Example: Predict marks for a new student
time_study = 5  
number_courses = 6  
marks_per_course = time_study / number_courses  

# Convert NumPy array to DataFrame with correct feature names
new_data = pd.DataFrame([[time_study, number_courses, marks_per_course]], 
                        columns=["time_study", "number_courses", "marks_per_course"])

predicted_marks = loaded_model.predict(new_data)

print(f"Predicted Marks: {predicted_marks[0]:.2f}")
