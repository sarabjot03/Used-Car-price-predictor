import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import pandas as pd
from scipy.sparse import hstack
import xgboost as xgb
import pickle

# Load the trained model and preprocessing objects from the pickle file
with open("/Users/sarabjotsingh/Desktop/Big Data Analytics/Term 2/Big Data Algos/xgb_model.pkl", "rb") as file:
    xgb_model, encoder, scaler = pickle.load(file)

# Define function for preprocessing user input and making predictions
def predict_price(year, make, model, trim, body, color, interior, odometer, mmr):
    # Create a DataFrame with user input
    user_data = pd.DataFrame({
        'year': [year],
        'make': [make],
        'model': [model],
        'trim': [trim],
        'body': [body],
        'color': [color],
        'interior': [interior],
        'odometer': [odometer],
        'mmr': [mmr]
    })

    # Perform one-hot encoding for textual features
    X_textual_encoded = encoder.transform(user_data[['make', 'model', 'trim', 'body', 'color', 'interior']])

    # Perform standard scaling for numerical features
    X_numerical_scaled = scaler.transform(user_data[['year', 'odometer', 'mmr']])

    # Combine textual and numerical features
    X = hstack((X_textual_encoded, X_numerical_scaled))

    # Make prediction
    prediction = xgb_model.predict(X)

    return prediction[0]

# Define function to handle button click event
def on_predict():
    try:
        # Retrieve user input
        year = int(year_entry.get())
        make = make_entry.get()
        model = model_entry.get()
        trim = trim_entry.get()
        body = body_entry.get()
        color = color_entry.get()
        interior = interior_entry.get()
        odometer = float(odometer_entry.get())
        mmr = float(mmr_entry.get())

        # Perform prediction
        predicted_price = predict_price(year, make, model, trim, body, color, interior, odometer, mmr)

        # Display predicted price
        messagebox.showinfo("Prediction", f"The predicted price is: ${predicted_price:.2f}")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

# Create tkinter window
window = tk.Tk()
window.title("Car Price Prediction")

# Create input fields
year_label = ttk.Label(window, text="Year:")
year_entry = ttk.Entry(window)
make_label = ttk.Label(window, text="Make:")
make_entry = ttk.Entry(window)
model_label = ttk.Label(window, text="Model:")
model_entry = ttk.Entry(window)
trim_label = ttk.Label(window, text="Trim:")
trim_entry = ttk.Entry(window)
body_label = ttk.Label(window, text="Body:")
body_entry = ttk.Entry(window)
color_label = ttk.Label(window, text="Color:")
color_entry = ttk.Entry(window)
interior_label = ttk.Label(window, text="Interior:")
interior_entry = ttk.Entry(window)
odometer_label = ttk.Label(window, text="Odometer:")
odometer_entry = ttk.Entry(window)
mmr_label = ttk.Label(window, text="MMR:")
mmr_entry = ttk.Entry(window)

# Create predict button
predict_button = ttk.Button(window, text="Predict", command=on_predict)

# Arrange input fields and button using grid layout
year_label.grid(row=0, column=0, padx=5, pady=5, sticky="e")
year_entry.grid(row=0, column=1, padx=5, pady=5)
make_label.grid(row=1, column=0, padx=5, pady=5, sticky="e")
make_entry.grid(row=1, column=1, padx=5, pady=5)
model_label.grid(row=2, column=0, padx=5, pady=5, sticky="e")
model_entry.grid(row=2, column=1, padx=5, pady=5)
trim_label.grid(row=3, column=0, padx=5, pady=5, sticky="e")
trim_entry.grid(row=3, column=1, padx=5, pady=5)
body_label.grid(row=4, column=0, padx=5, pady=5, sticky="e")
body_entry.grid(row=4, column=1, padx=5, pady=5)
color_label.grid(row=5, column=0, padx=5, pady=5, sticky="e")
color_entry.grid(row=5, column=1, padx=5, pady=5)
interior_label.grid(row=6, column=0, padx=5, pady=5, sticky="e")
interior_entry.grid(row=6, column=1, padx=5, pady=5)
odometer_label.grid(row=7, column=0, padx=5, pady=5, sticky="e")
odometer_entry.grid(row=7, column=1, padx=5, pady=5)
mmr_label.grid(row=8, column=0, padx=5, pady=5, sticky="e")
mmr_entry.grid(row=8, column=1, padx=5, pady=5)
predict_button.grid(row=9, column=0, columnspan=2, padx=5, pady=5)

# Start the tkinter event loop
window.mainloop()
