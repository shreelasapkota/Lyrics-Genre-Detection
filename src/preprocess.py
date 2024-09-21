import pandas as pd 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import re

nltk.download('stopwords') 


#Load and preprocess the CSV file data
def load_and_preprocess_csv_data(csv_file):
    df = pd.read_csv(csv_file)
    lyrics = df['lyrics']
    genres = df['type']
    stop_words = stopwords.words('english')
    def clean_lyrics(text):
        if isinstance(text,str):
            text = text.lower()
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            text = ''.join([char for char in text if char not in string.punctuation])
            text = ' '.join([word for word in text.split() if word not in stop_words and len(word) > 1]) 
            return text
        else:
            return '' #if the text is not string type
        
    df['cleaned_lyrics'] = lyrics.apply(clean_lyrics)
    df = df[df['cleaned_lyrics'].str.strip() != '']
    return df[['cleaned_lyrics', 'type']]

if __name__ == '__main__':
    csv_path = 'data/lyrics.csv'

    preprocessed_data = load_and_preprocess_csv_data(csv_path)
    print(preprocessed_data.head())

