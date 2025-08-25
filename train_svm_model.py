import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib

# Load data
df = pd.read_csv("handwriting_features.csv")
X = df.drop("label", axis=1)
y = df["label"]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = SVC(kernel='rbf', C=1, gamma='scale')
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("[📊] Classification Report:\n")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "svm_handwriting_model.pkl")
print("[✔] Model saved as svm_handwriting_model.pkl")
