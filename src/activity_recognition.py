import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 1: Load and preprocess the data from the CSV files
training_folder = "output"  # Replace with the actual path to the training folder
data = []

# Specify data types for the columns with mixed types
dtypes = {
    'time': float,
    'gFx': float,
    'gFy': float,
    'gFz': float,
    'ax': float,
    'ay': float,
    'az': float,
    'wx': float,
    'wy': float,
    'wz': float,
    'Azimuth': float,
    'Pitch': float,
    'Roll': float,
    'Latitude': float,
    'Longitude': float,
    'Speed (m/s)': float,
    'label': str
}

for filename in os.listdir(training_folder):
    if filename.endswith(".csv"):
        filepath = os.path.join(training_folder, filename)
        df = pd.read_csv(filepath, dtype=dtypes, low_memory=False)
        # Preprocess the data (e.g., cleaning, feature extraction, etc.)
        # Drop rows with 0 values in any column
        df = df.dropna(subset=df.columns[df.eq(0).any()])
        # Add the preprocessed data to the 'data' list
        data.append(df)

# Concatenate all the dataframes into a single DataFrame
df_combined = pd.concat(data, ignore_index=True)

# Extract features and labels
X = df_combined[['gFx', 'gFy', 'gFz', 'ax', 'ay', 'az', 'wx', 'wy', 'wz', 'Azimuth', 'Pitch', 'Roll', 'Latitude', 'Longitude', 'Speed (m/s)']].values
y = df_combined['label'].values

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Initialize the Random Forest classifier and train the model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Step 4: Evaluate the model's accuracy
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Replace 'new_file.csv' with the path to the new CSV file you want to predict activities for
new_file = pd.read_csv('walk2023-07-3117.41.55.csv')

# Preprocess the data in the new CSV file (similar to the preprocessing done during training)
# Drop rows with 0 values in any column
new_file = new_file.dropna(subset=new_file.columns[new_file.eq(0).any()])

# Extract features from the data
X_new = new_file[['gFx', 'gFy', 'gFz', 'ax', 'ay', 'az', 'wx', 'wy', 'wz', 'Azimuth', 'Pitch', 'Roll', 'Latitude', 'Longitude', 'Speed (m/s)']].values

# Use the trained model to predict activities in the new CSV file
y_pred_new = rf.predict(X_new)

# Print the predicted activities for each row in the new CSV file
# for activity in y_pred_new:
#     print("Predicted Activity:", activity)

# Add the predicted activities to the new CSV file in a new column "label_predict"
new_file['label_predict'] = y_pred_new

# Save the updated new CSV file with the "label_predict" column
new_file.to_csv('a1.csv', index=False)

#print("Predicted activities saved to 'walk2023-07-3117.41.55_predicted.csv'")