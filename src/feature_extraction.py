from sklearn.feature_extraction.text import TfidfVectorizer


def extract_features(lyrics):
    tfidf = TfidfVectorizer(max_features = 5000)
    X = tfidf.fit_transform(lyrics)

    return X, tfidf