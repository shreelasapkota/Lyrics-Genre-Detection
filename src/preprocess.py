import pandas as pd 
import nltk
from nltk.corpus import stopwords
import re

nltk.download('stopwords') 


#Cleaning the lyrics
def clean_lyrics(text):
    text=re.sub(r'd+','', text)  
    text = re.sub(r'[^\w\s]', '', text)
    text=text.lower()
    text=' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text

#Load and preprocess the CSV file data
def load_and_process_csv_data(csv_file):
    df = pd.read_csv(csv_file)
    df['clean_lyrics'] = df['lyrics'].apply(clean_lyrics)
    return df
