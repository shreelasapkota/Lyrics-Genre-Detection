import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from feature_extraction import extract_features_and_labels
from preprocess import load_and_preprocess_csv_data
import pickle

# def train_random_forest(X, y):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#  #random_test does the work of seed

#     model = RandomForestClassifier(n_estimators = 100, random_state=42)

#     model.fit(X_train, y_train)
    
#     return model, X_test, y_test

# if __name__ == '__main__':

   
#     data = load_and_preprocess_csv_data('data/lyrics.csv')
#     X_tfidf, y, _ = extract_features_and_labels(data)

#     model, X_test, y_test = train_random_forest(X_tfidf, y)
#     y_pred = model.predict(X_test)

#     print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
#     print("Classification Report:")
#     print(classification_report(y_test, y_pred))

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

















