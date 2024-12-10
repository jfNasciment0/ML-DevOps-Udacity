# Script to train machine learning model.
import os, pickle
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data

import logging

# Add the necessary imports for the starter code.
from ml.model import (
    train_model,
    compute_model_metrics,
    inference
)

# Initialize logging
logging.basicConfig(filename="fas.log", level=logging.INFO, filemode="a", format="%(name)s - %(levelname)s - %(message)s")

# Add code to load in the data.
model_dir = "../model/"

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

datapath = "../data/census.csv"
data = pd.read_csv(datapath)


# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(train, categorical_features=cat_features, label="salary", training=True)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

# Train and save a model.
model = train_model(X_train, y_train)

# Save the trained model, encoder, and labelizer to the specified paths
# joblib.dump(model, os.path.join(model_dir, "trained_model.pkl"))
# joblib.dump(encoder, os.path.join(model_dir, "encoder.pkl"))
# joblib.dump(lb, os.path.join(model_dir, "labelizer.pkl"))

# if saved model exits, load the model from disk
if os.path.isfile(os.path.join(model_dir, "trained_model.pkl")):
        model = pickle.load(open(os.path.join(model_dir, "trained_model.pkl"), 'rb'))
        encoder = pickle.load(open(os.path.join(model_dir, "encoder.pkl"), 'rb'))
        lb = pickle.load(open(os.path.join(model_dir, "labelizer.pkl"), 'rb'))

# Else Train and save a model.
else:
    model = train_model(X_train, y_train)
    # save model  to disk in ./model folder
    pickle.dump(model, open(os.path.join(model_dir, "trained_model.pkl"), 'wb'))
    pickle.dump(encoder, open(os.path.join(model_dir, "encoder.pkl"), 'wb'))
    pickle.dump(lb, open(os.path.join(model_dir, "labelizer.pkl"), 'wb'))
    logging.info(f"Model saved to disk: {model_dir}")

logging.info(f"The Model was saved: {model_dir}")

inference_predictions = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, inference_predictions)

logging.info(f"Classification target labels: {list(lb.classes_)}")
logging.info(f"Precision:{precision:.3f}, Recall:{recall:.3f}, Fbeta:{fbeta:.3f}")
