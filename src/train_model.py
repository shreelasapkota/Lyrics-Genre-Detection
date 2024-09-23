import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from feature_extraction import extract_features_and_labels
from preprocess import load_and_preprocess_csv_data
import pickle


def train_random_forest(X, y):
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    
    # Save the model
    with open('models/random_forest_classifier.pkl', 'wb') as f:
        pickle.dump(model, f)

    return model

if __name__ == '__main__':
    data = load_and_preprocess_csv_data('data/lyrics.csv')
    X_tfidf, y, _ = extract_features_and_labels(data)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

    # Train the model
    model = train_random_forest(X_train, y_train)

    # Optional: Evaluate on the training set
    print("Model trained and saved.")

















