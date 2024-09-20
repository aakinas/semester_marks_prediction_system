import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Initialize the tkinter window
window = tk.Tk()
window.title("End Semester Marks Prediction System")
window.geometry("1200x600")  # Adjust window size
window.configure(bg="#F0F0F0")  # Set background color

# Load the dataset (replace 'Marks.csv' with your dataset file)
data = pd.read_csv('Marks.csv')

# Preprocess the data
data.fillna(data.mean(), inplace=True)  # Impute missing values with mean
X = data[['dbms_t1', 'dbms_t2', 'ML_t1', 'ML_t2', 'TOC_t1', 'TOC_t2', 'OS_t1', 'OS_t2', 'CS_t1', 'CS_t2']]
y = data[['dbms_ese', 'ML_ese', 'TOC_ese', 'OS_ese', 'CS_ese']]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Regression model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'trained_model.pkl')

# Define grade and credit hour mappings for each subject
grades = {'A+': 10, 'A': 9, 'B+': 8, 'B': 7, 'C': 6, 'D': 5, 'F': 0}
credit_hours = {'DBMS': 4, 'ML': 3, 'TOC': 3, 'OS': 3, 'CS': 4}

def get_grade(marks):
    if marks >= 90:
        return 'A+'
    elif 80 <= marks < 90:
        return 'A'
    elif 70 <= marks < 80:
        return 'B+'
    elif 60 <= marks < 70:
        return 'B'
    elif 50 <= marks < 60:
        return 'C'
    elif 40 <= marks < 50:
        return 'D'
    else:
        return 'F'
    
list1=['DBMS', 'ML', 'TOC', 'OS', 'CS']

def calculate_sgpa(predictions, test1_marks, test2_marks):
    total_marks = {}
    for i, subject in enumerate(['DBMS', 'ML', 'TOC', 'OS', 'CS']):
        total_marks[subject] = predictions[i] + test1_marks[i] + test2_marks[i]

    grade_points = sum(grades[get_grade(total_marks[subject])] * credit_hours[subject] for subject in total_marks)
    total_credit_hours = sum(credit_hours[subject] for subject in total_marks)
    sgpa = grade_points / total_credit_hours
    return sgpa

def show_prediction():
    def predict_marks():
        try:
            # Get user inputs
            test1_marks = [float(test1_entries[i].get()) for i in range(5)]
            test2_marks = [float(test2_entries[i].get()) for i in range(5)]

            # Check for negative numbers
            if any(mark < 0 for mark in test1_marks) or any(mark < 0 for mark in test2_marks):
                messagebox.showerror("Error", "Marks cannot be negative.")
                return

            # Check for marks exceeding 25
            if any(mark > 25 for mark in test1_marks) or any(mark > 25 for mark in test2_marks):
                messagebox.showerror("Error", "Marks cannot exceed 25.")
                return

            # Load the trained model
            model = joblib.load('trained_model.pkl')

            # Make prediction
            input_data = [test1_marks + test2_marks]
            prediction = model.predict(input_data)[0]

            # Display prediction
            predicted_marks_label.config(text=f'Predicted End Semester Marks: {prediction}', foreground="#006400", font=("Arial", 14))

            # Calculate SGPA
            sgpa = calculate_sgpa(prediction, test1_marks, test2_marks)

            # Display SGPA
            sgpa_label.config(text=f'Predicted SGPA: {sgpa:.2f}', foreground="#006400", font=("Arial", 14))

            # Create scatter plot
            fig, ax = plt.subplots()
            ax.scatter(list1, prediction, color='blue')
            ax.set_xlabel('Subjects')
            ax.set_ylabel('Predicted ESE Marks')
            ax.set_title('Predicted Marks Scatter Plot')

            # Clear previous plot
            for widget in plot_frame.winfo_children():
                widget.destroy()

            canvas = FigureCanvasTkAgg(fig, master=plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack()

        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric values for all subjects.")

    # Create prediction window
    prediction_window = tk.Toplevel(window)
    prediction_window.title("Predict End Semester Marks and SGPA")
    prediction_window.geometry("900x600")  # Adjust prediction window size
    prediction_window.configure(bg="MistyRose1")  # Set background color for prediction window

    # Create input fields for each subject's test1 and test2 marks in the prediction window
    subjects = ['DBMS', 'ML', 'TOC', 'OS', 'CS']
    test1_entries = []
    test2_entries = []

    for subject in subjects:
        test1_label = ttk.Label(prediction_window, text=f"{subject} Test 1 Marks:", font=("Arial", 12), foreground="#8B008B", background="MistyRose1")
        test1_label.grid(row=subjects.index(subject), column=0, pady=5)
        test1_entry = ttk.Entry(prediction_window, font=("Arial", 12))
        test1_entry.grid(row=subjects.index(subject), column=1)
        test1_entries.append(test1_entry)

        test2_label = ttk.Label(prediction_window, text=f"{subject} Test 2 Marks:", font=("Arial", 12), foreground="#8B008B", background="MistyRose1")
        test2_label.grid(row=subjects.index(subject), column=2, pady=5)
        test2_entry = ttk.Entry(prediction_window, font=("Arial", 12))
        test2_entry.grid(row=subjects.index(subject), column=3)
        test2_entries.append(test2_entry)

    # Create a button to trigger prediction
    predict_button = ttk.Button(prediction_window, text="Predict", command=predict_marks)
    predict_button.grid(row=len(subjects), column=1, pady=10, columnspan=2,padx=20)

    # Create labels to display the predicted result and SGPA
    predicted_marks_label = ttk.Label(prediction_window, text="", font=("Arial", 12), foreground="#006400")
    predicted_marks_label.grid(row=len(subjects) + 1, column=1, pady=5, columnspan=2)

    sgpa_label = ttk.Label(prediction_window, text="", font=("Arial", 12), foreground="#006400")
    sgpa_label.grid(row=len(subjects) + 2, column=1, pady=5, columnspan=2)

    # Create a frame to hold the scatter plot
    global plot_frame
    plot_frame = ttk.Frame(window)
    plot_frame.grid(row=0, column=1, rowspan=len(subjects) + 3, padx=20, pady=20, sticky="nsew")

# Create login window elements
window.configure(background="MistyRose1")

login_label = ttk.Label(window, text="Please click to continue", font=("Arial", 16, "bold"), foreground="#8B008B", background="MistyRose1")
login_label.grid(row=0, column=0, pady=30,padx=20)

style = ttk.Style()
style.configure("Accent.TButton", foreground="MistyRose1", background="MistyRose1", font=("Arial", 12, "bold"))

login_button = ttk.Button(window, text="click here", command=show_prediction)
login_button.grid(row=1, column=0, pady=20,padx=20)

# Add section for project details
project_label = ttk.Label(window, text="Project Details", font=("Arial", 14, "bold"), foreground="#8B008B", background="MistyRose1")
project_label.grid(row=2, column=0, pady=10,padx=20)

# Add your project details here
project_details = ttk.Label(window, text="Project Name:prediction of Marks and SGPA\nLanguage: Python\nTools: tkinter, Machine learning model-RandomForest\nDataset:Marks.csv", font=("Arial", 12), foreground="black", background="MistyRose1")
project_details.grid(row=3, column=0, pady=10,padx=20)

# Start the GUI event loop
window.mainloop()
