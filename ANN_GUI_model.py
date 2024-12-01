import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import mean_absolute_error, r2_score
import tkinter as tk
from tkinter import filedialog, messagebox

# Train the model (this should be done separately once and saved as 'your_model.h5')
# Load your data and train the model (same as before)
data = pd.read_csv(r'D:\Sir AB\Data\Correctdata.csv')

# Separate input features (X) and target variable (Y)
Input = data.iloc[:, :-1].values
Target = data.iloc[:, -1].values

# Transpose to match input/output shape if necessary
x = Input.T
t = Target.T

# Split the data into training, validation, and test sets
train_size = int(0.8 * x.shape[1])
val_size = int(0.1 * x.shape[1])
x_train, x_val, x_test = x[:, :train_size], x[:, train_size:train_size+val_size], x[:, train_size+val_size:]
t_train, t_val, t_test = t[:train_size], t[train_size:train_size+val_size], t[train_size+val_size:]

# Define the ANN model
model = models.Sequential()
model.add(layers.Input(shape=(x.shape[0],)))
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1))

# Compile the model
model.compile(optimizer='adam', 
              loss='mean_squared_error',
              metrics=['accuracy'])

# Train the model
model.fit(x_train.T, t_train.T, epochs=200, batch_size=10, validation_data=(x_val.T, t_val.T))

# Save the trained model
model.save('your_model.h5')  # Save the trained model for future use

# ------------------------------------------

# Load the trained ANN model
model = tf.keras.models.load_model('your_model.h5')

# Validate inputs
def validate_inputs():
    valid = True
    for entry in input_entries:
        try:
            float(entry.get())
            entry.config(bg="white")  # Reset to white for valid input
        except ValueError:
            entry.config(bg="red")  # Highlight invalid fields in red
            valid = False
    return valid

# Predict based on inputs using the trained model
def predict_input():
    if not validate_inputs():
        messagebox.showerror("Error", "Invalid inputs! Please correct highlighted fields.")
        return
    
    inputs = [float(entry.get()) for entry in input_entries]
    
    # Reshape input data for model prediction
    inputs = np.array(inputs).reshape(1, -1)
    
    # Make prediction using the trained model
    prediction = model.predict(inputs)
    
    # Show the predicted NCDE
    result_label.config(text=f"Predicted NCDE: {prediction[0][0]:.2f}")
    
    # Log inputs and prediction
    log_text.insert(tk.END, f"Inputs: {inputs.flatten()} -> Prediction (NCDE): {prediction[0][0]:.2f}\n")

# Save inputs
def save_inputs():
    inputs = {label: entry.get() for label, entry in zip(input_labels, input_entries)}
    file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
    if file_path:
        pd.DataFrame([inputs]).to_csv(file_path, index=False)
        messagebox.showinfo("Saved", "Inputs saved successfully.")

# Load inputs
def load_inputs():
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        data = pd.read_csv(file_path)
        for i, (label, entry) in enumerate(zip(input_labels, input_entries)):
            entry.delete(0, tk.END)
            entry.insert(0, str(data.iloc[0][label]))
        messagebox.showinfo("Loaded", "Inputs loaded successfully.")

# Export logs
def export_logs():
    file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
    if file_path:
        with open(file_path, "w") as file:
            file.write(log_text.get("1.0", tk.END))
        messagebox.showinfo("Exported", "Logs exported successfully.")

# Clear logs
def clear_logs():
    log_text.delete("1.0", tk.END)

# Show help
def show_help():
    help_message = """
    Input Field Descriptions:
    - lw: Wall length (mm)
    - hw: Wall height (mm)
    - tw: Wall thickness (mm)
    - f'c: Concrete compressive strength (MPa)
    - fyt: Yield strength of tension reinforcement (MPa)
    - fysh: Yield strength of shear reinforcement (MPa)
    - fyl: Yield strength of longitudinal reinforcement (MPa)
    - fybl: Yield strength of boundary layer reinforcement (MPa)
    - ρt: Reinforcement ratio for tension
    - ρsh: Shear reinforcement ratio
    - ρl: Longitudinal reinforcement ratio
    - ρbl: Boundary layer reinforcement ratio
    - P/(Agf′c): Axial load ratio
    - b0: Boundary layer width (mm)
    - db: Diameter of reinforcement (mm)
    - s/db: Spacing-to-diameter ratio
    - AR: Aspect ratio
    - M/Vlw: Moment-to-shear ratio
    """
    messagebox.showinfo("Help/Info", help_message)

# GUI Setup
root = tk.Tk()
root.title("RC Shear Wall Energy Dissipation Prediction")

# Input field labels (full names with units)
input_labels = [
    "Wall Length (lw, mm)", 
    "Wall Height (hw, mm)", 
    "Wall Thickness (tw, mm)", 
    "Concrete Compressive Strength (f′c, MPa)", 
    "Yield Strength of Tension Reinforcement (fyt, MPa)", 
    "Yield Strength of Shear Reinforcement (fysh, MPa)", 
    "Yield Strength of Longitudinal Reinforcement (fyl, MPa)", 
    "Yield Strength of Boundary Layer Reinforcement (fybl, MPa)", 
    "Reinforcement Ratio for Tension (ρt)", 
    "Shear Reinforcement Ratio (ρsh)", 
    "Longitudinal Reinforcement Ratio (ρl)", 
    "Boundary Layer Reinforcement Ratio (ρbl)", 
    "Axial Load Ratio (P/(Agf′c))", 
    "Boundary Layer Width (b0, mm)", 
    "Reinforcement Diameter (db, mm)", 
    "Spacing-to-Diameter Ratio (s/db)", 
    "Aspect Ratio (AR)", 
    "Moment-to-Shear Ratio (M/Vlw)"
]

# Input fields
input_entries = []
for i, label in enumerate(input_labels):
    tk.Label(root, text=label).grid(row=i, column=0, padx=10, pady=5, sticky="w")
    entry = tk.Entry(root, width=20)
    entry.grid(row=i, column=1, padx=10, pady=5)
    input_entries.append(entry)

# Buttons
button_frame = tk.Frame(root)
button_frame.grid(row=len(input_labels), column=0, columnspan=2, pady=10)

predict_button = tk.Button(button_frame, text="Predict", command=predict_input)
predict_button.grid(row=0, column=0, padx=5)

save_button = tk.Button(button_frame, text="Save Inputs", command=save_inputs)
save_button.grid(row=0, column=1, padx=5)

load_button = tk.Button(button_frame, text="Load Inputs", command=load_inputs)
load_button.grid(row=0, column=2, padx=5)

export_button = tk.Button(button_frame, text="Export Logs", command=export_logs)
export_button.grid(row=0, column=3, padx=5)

clear_button = tk.Button(button_frame, text="Clear Logs", command=clear_logs)
clear_button.grid(row=0, column=4, padx=5)

help_button = tk.Button(button_frame, text="Help/Info", command=show_help)
help_button.grid(row=0, column=5, padx=5)

# Result Label
result_label = tk.Label(root, text="Predicted NCDE: N/A", font=("Arial", 12))
result_label.grid(row=len(input_labels) + 1, column=0, columnspan=2, pady=10)

# Logs
tk.Label(root, text="Logs:").grid(row=len(input_labels) + 2, column=0, padx=10, sticky="w")
log_text = tk.Text(root, height=10, width=50)
log_text.grid(row=len(input_labels) + 3, column=0, columnspan=2, padx=10, pady=5)

# Run GUI
root.mainloop()
