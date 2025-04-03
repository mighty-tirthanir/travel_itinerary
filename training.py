import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# --- Load Dataset ---
df = pd.read_csv('Final_Updated_Review_db3.csv', dtype=str, low_memory=False)

# Drop missing values
df = df.dropna(subset=['City', 'Place', 'Review', 'Rating'])

# Convert ratings to numerical values
df['Rating'] = df['Rating'].astype(float)

# --- Feature Extraction using TF-IDF ---
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
X = vectorizer.fit_transform(df['Review'])

# Target variable (ratings)
y = df['Rating']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train XGBoost Model ---
model = LogisticRegression()
model.fit(X_train, y_train)

# --- Model Evaluation ---
y_pred = model.predict(X_test)


# Calculate Errors
mae = mean_absolute_error(y_test, y_pred)  # Mean Absolute Error
mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error
rmse = np.sqrt(mse)  # Root Mean Squared Error
r2 = r2_score(y_test, y_pred)  # R² Score (Accuracy metric for regression)

# Print Evaluation Metrics
print(f"✅ Model Evaluation Results:")
print(f"🔹 Mean Absolute Error (MAE): {mae:.4f}")
print(f"🔹 Mean Squared Error (MSE): {mse:.4f}")
print(f"🔹 Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"🔹 R² Score: {r2:.4f}")  # Higher is better, max = 1

# --- Save Model and Vectorizer ---
with open("location_recommendation_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\n✅ Model and vectorizer saved successfully!")
train_sizes = np.linspace(0.1, 0.9, 10)  # Different train-test splits
accuracies = []
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
accuracies.append(accuracy)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, accuracies, marker='o', linestyle='-', color='b', label="Accuracy")

plt.xlabel("Training Data Percentage")
plt.ylabel("Accuracy Score")
plt.title("Logistic Regression Accuracy vs Train-Test Split")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
