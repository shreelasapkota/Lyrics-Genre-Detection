from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

def train_random_forest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=2, random_test=42) #random_test does the work of seed

    model = RandomForestClassifier(n_estimators = 100, random_state=42)

    model.fit(X_train, y_train)

    with open('models/random_forest_model.pkl', 'wb') as f:
        pickle.dump(model,f)
    
    #OR pickle.dump(model, open('random_forest_model.pkl', 'wb'))
    return model, X_test, y_test























