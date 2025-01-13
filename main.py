import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('FinancialMarketData.csv')

# Preprocessing (as per the steps you did in Colab)
data = data.fillna(method='ffill')  # Handle missing values

# Cleaning the dataset by dropping fully empty columns and rows
data_cleaned = data.dropna(how='all').reset_index(drop=True)
data_cleaned = data_cleaned.dropna(axis=1, how='all')

# Attempt to infer numerical columns by converting to numeric (coercing errors)
numeric_cols = []
for col in data_cleaned.columns:
    try:
        pd.to_numeric(data_cleaned[col], errors='raise')
        numeric_cols.append(col)
    except ValueError:
        continue

# Display cleaned dataset sample and numerical columns identified
data_cleaned.head(), numeric_cols

# Setting row 3 as the header and reloading the dataset from there
data_cleaned.columns = data_cleaned.iloc[3]  # Set the 4th row as header (index 3)
data_cleaned = data_cleaned[4:].reset_index(drop=True)  # Drop rows above the data

# Attempting to detect numerical columns again after cleanup
numeric_cols = []
for col in data_cleaned.columns:
    try:
        pd.to_numeric(data_cleaned[col], errors='raise')
        numeric_cols.append(col)
    except ValueError:
        continue

# Displaying a preview of the cleaned dataset and identified numerical columns
data_cleaned.head(), numeric_cols

from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Initializing encoders and scalers
label_encoder = LabelEncoder()
scaler = MinMaxScaler()

# Encoding categorical variables and normalizing numerical features
encoded_data = data_cleaned.copy()

# Encoding categorical columns
for col in encoded_data.select_dtypes(include=['object']).columns:
    try:
        encoded_data[col] = label_encoder.fit_transform(encoded_data[col].astype(str))
    except Exception as e:
        pass

# Normalizing numerical columns
for col in encoded_data.select_dtypes(include=['int64', 'float64']).columns:
    encoded_data[col] = scaler.fit_transform(encoded_data[[col]])

# Display the first few rows of the processed dataset
encoded_data.head()

# Inspect the dataset to identify potential label columns
print("Columns in dataset:", data.columns)
print("Sample data:\n", data.head())

label_column_name = 'Fixed Income, MM & Interbank'  # Specify the label column
if label_column_name not in data.columns:
    raise ValueError(f"Label column '{label_column_name}' not found in dataset.")

# Split data into features (X) and labels (y)
x = data.drop(columns=[label_column_name])  # All columns except the label
y = data[label_column_name]  # The target column

# Display shapes and sample data
print("Features shape:", x.shape)
print("Labels shape:", y.shape)
print("Features (X) sample:\n", x.head())
print("Labels (y) sample:\n", y.head())

# Optional: Save the features and labels to separate files
x.to_csv('features.csv', index=False)
y.to_csv('labels.csv', index=False)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Inspect dataset and columns
print("Columns in dataset:", data.columns)

# Set the label column (replace 'YourLabelColumnName' with the actual name)
label_column_name = 'Fixed Income, MM & Interbank'  # Replace this with the correct label column

# Splitting the data into features (X) and labels (y)
X = data.drop(columns=[label_column_name])
y = data[label_column_name]

# Encoding categorical variables if any (e.g., 'object' type columns)
label_encoder = LabelEncoder()
for col in X.select_dtypes(include=['object']).columns:
    X[col] = label_encoder.fit_transform(X[col].astype(str))

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Output the results
print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

joblib.dump(model, 'models/')