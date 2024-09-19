import pandas as pd 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

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
            text = ''.join([char for char in text if char not in string.punctuation])
            text = ' '.join([word for word in text if word not in stop_words ]) 
            return text
        else:
            return '' #if the text is not string type
        
    df['cleaned_lyrics'] = lyrics.apply(clean_lyrics)
    return df[['cleaned_lyrics', 'type']]

if __name__ == '__main__':
    csv_path = 'data/lyrics.csv'

    preprocessed_data = load_and_preprocess_csv_data(csv_path)
    print(preprocessed_data.head())

