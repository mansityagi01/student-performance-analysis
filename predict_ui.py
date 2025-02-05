import joblib  # Use joblib instead of pickle
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox

# Load the trained model using joblib (same as your working script)
loaded_model = joblib.load("student_marks_predictor.pkl")

# Create Tkinter UI
root = tk.Tk()
root.title("Student Performance Predictor")

tk.Label(root, text="Enter Study Time (hours):").pack()
entry_study_time = tk.Entry(root)
entry_study_time.pack()

tk.Label(root, text="Enter Number of Courses:").pack()
entry_courses = tk.Entry(root)
entry_courses.pack()

tk.Label(root, text="Enter Marks Per Course:").pack()
entry_marks = tk.Entry(root)
entry_marks.pack()

def predict_marks():
    try:
        # Get user input
        time_study = float(entry_study_time.get())
        number_courses = int(entry_courses.get())
        marks_per_course = float(entry_marks.get())

        # Create DataFrame for model input
        new_data = pd.DataFrame([[time_study, number_courses, marks_per_course]], 
                                columns=["time_study", "number_courses", "marks_per_course"])

        # Predict
        predicted_marks = loaded_model.predict(new_data)[0]
        messagebox.showinfo("Prediction Result", f"Predicted Marks: {predicted_marks:.2f}")

    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values.")

tk.Button(root, text="Predict", command=predict_marks).pack()
root.mainloop()
