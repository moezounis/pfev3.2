import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

# Load data from CSV file
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        print("Loaded data:")
        print(data.head())  # Print the first few rows of the DataFrame
        return data
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None

# Train and evaluate the model
def train_and_evaluate_model(X, y):
    model = RandomForestClassifier()
    param_grid = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [5, 10, 15, 20]
    }
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    grid_search.fit(X, y)
    print(f"Best parameters: {grid_search.best_params_}")
    model = RandomForestClassifier(n_estimators=grid_search.best_params_['n_estimators'],
                                   max_depth=grid_search.best_params_['max_depth'])
    model.fit(X, y)
    return model

# Load and preprocess data
def load_and_preprocess_data(file_path):
    data = load_data(file_path)
    if data is not None:
        # Split data into features and target
        X = data.drop('label', axis=1)
        y = data['label']
        return X, y
    return None, None

# Function to get pre-trained model
def get_trained_model(data_file_path):
    X, y = load_and_preprocess_data(data_file_path)
    if X is not None and y is not None:
        model = train_and_evaluate_model(X, y)
        return model
    else:
        return None
