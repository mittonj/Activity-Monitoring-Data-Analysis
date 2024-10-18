import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


def load_data(folder_path):
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            filepath = os.path.join(folder_path, filename)
            df = pd.read_csv(filepath)
            data.append(df)
    return pd.concat(data, ignore_index=True)

def preprocess_data(data):
    X = data.drop(columns=['activity'])
    y = data['activity']
    return X, y

def train_RandomForest(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

def train_NeuralNets(X_train, y_train):
    model = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(14,3))
    model.fit(X_train, y_train)
    return model

def makePredictions(model, testing_data, test_folder, X_train, y_train, X_valid, y_valid, X_test, y_test):
    
    # Evaluate accuracy on validation set
    print(f"Accuracy on validation set: {model.score(X_valid, y_valid):.2f}")

    # Preprocess testing data (excluding 'activity' column)

    # Make predictions
    predictions = model.predict(X_test)

    # Evaluate accuracy on testing data
    print(f"Accuracy on testing data: {model.score(X_test, y_test):.2f}")

    # Create a folder for the model if it doesn't exist
    model_output_folder = os.path.join(os.path.dirname(__file__), 'output_pd', model.__class__.__name__)
    if not os.path.exists(model_output_folder):
        os.makedirs(model_output_folder)

    # Add predictions to the testing data and save to the model folder
    testing_data['prediction'] = predictions
    for filename in os.listdir(test_folder):
        if filename.endswith(".csv"):
            filepath = os.path.join(test_folder, filename)
            output_filepath = os.path.join(model_output_folder, filename)
            df = pd.read_csv(filepath)
            df['prediction'] = testing_data.loc[df.index, 'prediction']
            df.to_csv(output_filepath, index=False)


def main():
    # Check if the correct number of command-line arguments are provided
    if len(sys.argv) != 3:
        print("Usage: python3 activity_recognition.py <training_folder> <test_folder>")
        sys.exit(1)

    training_folder = sys.argv[1]
    test_folder = sys.argv[2]

    # Load training and testing data
    training_data = load_data(training_folder)
    testing_data = load_data(test_folder)

    training_data = training_data.drop(columns=['filename'])
    testing_data = testing_data.drop(columns=['filename'])

    # Preprocess training data
    X_train, y_train = preprocess_data(training_data)
    X_test, y_test = preprocess_data(testing_data)


    print(X_train.columns)

    # Train-test split for validation
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.3, random_state=1)


    models = []

    # Train the model using RandomForestClassifier
    randomForest_model = train_RandomForest(X_train, y_train)
    MLP_model = train_NeuralNets(X_train, y_train)


    models.append(randomForest_model)
    models.append(MLP_model)

    for model in models:
        makePredictions(model, testing_data, test_folder, X_train, y_train, X_valid, y_valid, X_test, y_test)

    print("Predictions saved in the 'output_pd' folder.")

if __name__ == "__main__":
    main()
