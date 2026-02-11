# File: src/model_development.py
import os
import pickle
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

WORKING_DIR = "/opt/airflow/working_data"
MODEL_DIR = "/opt/airflow/model"
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data() -> str:
    """
    Load CSV and persist raw dataframe to a pickle file.
    Returns path to saved file.
    """
    csv_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "advertising.csv",
    )
    df = pd.read_csv(csv_path)

    out_path = os.path.join(WORKING_DIR, "raw.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(df, f)
    return out_path

def data_preprocessing(file_path: str) -> str:
    """
    Load dataframe, split, scale, and save (X_train, X_test, y_train, y_test) to pickle.
    Returns path to saved file.
    """
    with open(file_path, "rb") as f:
        df = pickle.load(f)

    X = df.drop(
        ["Timestamp", "Clicked on Ad", "Ad Topic Line", "Country", "City"],
        axis=1,
    )
    y = df["Clicked on Ad"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    num_columns = [
        "Daily Time Spent on Site",
        "Age",
        "Area Income",
        "Daily Internet Usage",
        "Male",
    ]

    ct = make_column_transformer(
        (MinMaxScaler(), num_columns),
        (StandardScaler(), num_columns),
        remainder="passthrough",
    )

    X_train_tr = ct.fit_transform(X_train)
    X_test_tr = ct.transform(X_test)

    out_path = os.path.join(WORKING_DIR, "preprocessed.pkl")
    with open(out_path, "wb") as f:
        pickle.dump((X_train_tr, X_test_tr, y_train.values, y_test.values), f)
    return out_path

def separate_data_outputs(file_path: str) -> str:
    """
    Passthrough; kept so the DAG composes cleanly.
    """
    return file_path

def build_model(file_path: str, filename: str) -> str:
    """
    Train LR model and save to MODEL_DIR/filename. Returns model path.
    """
    with open(file_path, "rb") as f:
        X_train, X_test, y_train, y_test = pickle.load(f)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    model_path = os.path.join(MODEL_DIR, filename)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    return model_path

def load_model(file_path: str, filename: str) -> int:
    """
    Load saved model and test set, print score, and return first prediction as int.
    """
    with open(file_path, "rb") as f:
        X_train, X_test, y_train, y_test = pickle.load(f)

    model_path = os.path.join(MODEL_DIR, filename)
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    score = model.score(X_test, y_test)
    print(f"Model score on test data: {score}")

    pred = model.predict(X_test)
    return int(pred[0])


def get_model_accuracy(file_path: str, filename: str) -> float:
    """
    Calculate and return the model's accuracy on test data.
        
    Args:
        file_path: Path to the pickle file containing (X_train, X_test, y_train, y_test)
        filename: Name of the saved model file (e.g., "model.sav")
    
    Returns:
        float: Accuracy score between 0.0 (0% correct) and 1.0 (100% correct)
    """
    # Load the test data
    with open(file_path, "rb") as f:
        X_train, X_test, y_train, y_test = pickle.load(f)

    # Load the trained model
    model_path = os.path.join(MODEL_DIR, filename)
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Calculate accuracy: percentage of correct predictions
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy:.2%}")  # e.g., "95.00%"
    
    return accuracy


def check_accuracy_threshold(accuracy: float, threshold: float = 0.8) -> bool:
    """
    Check if model accuracy meets the required threshold.

    Args:
        accuracy: The model's accuracy score (0.0 to 1.0)
        threshold: Minimum required accuracy (default 0.8 = 80%)
    
    Returns:
        bool: True if accuracy >= threshold, False otherwise
    """
    passed = accuracy >= threshold
    
    if passed:
        print(f"✓ Model PASSED: {accuracy:.2%} >= {threshold:.0%}")
    else:
        print(f"✗ Model FAILED: {accuracy:.2%} < {threshold:.0%}")
    
    return passed
