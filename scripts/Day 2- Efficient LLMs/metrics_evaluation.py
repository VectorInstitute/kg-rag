import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Load the dataset
df = pd.read_csv('llama3.2.csv')

# Extract and standardize the case for actual and predicted labels
y_true = df['label'].str.lower()
y_pred = df['predicted'].str.lower()

# Calculate precision, recall, F1-score, and accuracy
precision = precision_score(y_true, y_pred, average='weighted', zero_division=1)
recall = recall_score(y_true, y_pred, average='weighted', zero_division=1)
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=1)
accuracy = accuracy_score(y_true, y_pred)

# Print the results
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Accuracy:", accuracy)
