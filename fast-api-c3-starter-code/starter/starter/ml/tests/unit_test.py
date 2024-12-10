import pytest
import pandas as pd
import joblib
import numpy as np
from model import train_model, compute_model_metrics, inference
from sklearn.model_selection import train_test_split
from starter.starter.ml.data import process_data  # Assuming process_data is defined here

# Path to the CSV file for testing
DATA_PATH = "../data/census.csv"


@pytest.fixture
def setup_data():
    # Load data from CSV
    data = pd.read_csv(DATA_PATH)

    # Specify categorical features and label column
    cat_features = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]

    # Split the data into training and testing sets
    train, test = train_test_split(data, test_size=0.2, random_state=42)

    # Process the training and test data
    X_train, y_train, encoder, lb = process_data(train, categorical_features=cat_features, label="salary", training=True)
    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )

    return X_train, X_test, y_train, y_test, encoder, lb


def test_train_model(setup_data):
    X_train, _, y_train, _, _, _ = setup_data
    model = train_model(X_train, y_train)
    assert model is not None
    assert hasattr(model, "predict")


def test_compute_model_metrics(setup_data):
    X_train, X_test, y_train, y_test, encoder, lb = setup_data
    model = train_model(X_train, y_train)
    predictions = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, predictions)

    # Ensure that the metrics are float values
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)


def test_inference(setup_data):
    X_train, X_test, y_train, y_test, _, _ = setup_data
    model = train_model(X_train, y_train)
    predictions = inference(model, X_test)

    assert len(predictions) == len(y_test)
    assert np.all(np.isin(predictions, [0, 1]))  # Assuming binary classification


def test_model_loading():
    # Load the saved model, encoder, and labelizer
    model = joblib.load("../model/trained_model.pkl")
    encoder = joblib.load("../model/encoder.pkl")
    lb = joblib.load("../model/labelizer.pkl")

    assert model is not None
    assert encoder is not None
    assert lb is not None
    assert hasattr(model, "predict")
