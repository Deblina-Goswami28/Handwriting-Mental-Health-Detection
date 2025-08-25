import joblib
from extract_features import extract_features_from_image
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import numpy as np

# Load model
model = joblib.load("svm_handwriting_model.pkl")

# Select file
Tk().withdraw()
img_path = askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
if not img_path:
    print("No file selected.")
    exit()

# Extract features
features = extract_features_from_image(img_path)
if not features:
    print("Could not extract features.")
    exit()

# Predict
features_np = np.array(features).reshape(1, -1)
prediction = model.predict(features_np)[0]

print("\n--- Result ---")
print(f"Image: {img_path}")
print(f"Predicted Class: {prediction}")
