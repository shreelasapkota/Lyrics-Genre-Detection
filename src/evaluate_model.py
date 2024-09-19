from sklearn.metrics import classification_report, confusion_matrix, accuracyscore
import pickle

#evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracyscore(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    print("classification report:")
    print(classification_report(y_test, y_pred))

    print("Confusion matrix: ")
    print(confusion_matrix(y_test, y_pred))


#load and evaluate the model
def load_and_evaluate_model(X_test, y_pred):
    #load the model
    with open('models/random_forest_classifier.pkl', 'rb') as s:
        model = pickle.load(s) 

    #evaluate the model
    evaluate_model(model, X_test, y_pred)

if __name__ == '__main__':
    from feature_extraction import extract_features_and_labels, split_data
    from preprocess import load_and_preprocess_csv_data

    data = load_and_preprocess_csv_data('data/lyrics.csv')
    X_tfidf, y, _ = extract_features_and_labels(data)
    X_train, X_test, y_train, y_test = split_data(X_tfidf, y)
    load_and_evaluate_model(X_test, y_test)