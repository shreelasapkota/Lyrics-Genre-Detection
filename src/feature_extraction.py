from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def extract_features_and_labels(df):

    X = df['cleaned_lyrics']
    y = df['type']
    print(X.head())
    
    tfidf = TfidfVectorizer(max_features = 5000)
    X_tfidf = tfidf.fit_transform(X)
    
    return X_tfidf,y, tfidf

def split_data(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=42)
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    from preprocess import load_and_preprocess_csv_data
    data = load_and_preprocess_csv_data('data/lyrics.csv')
    X_tfidf, y, tfidf = extract_features_and_labels(data)
    X_train, X_test, y_train, y_test = split_data(X_tfidf, y)
    print(X_train.shape, X_test.shape)
