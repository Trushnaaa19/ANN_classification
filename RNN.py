import pandas as pd

df = pd.read_csv(r"C:\Users\LENOVO\Downloads\Telegram Desktop\02.IMDB Dataset.csv")
df.head()
nullvalue = df.isnull().sum()
print(nullvalue)
df.drop_duplicates(inplace = True)

#pre-processing
 
#converting to lowercase 
df["review"] = df["review"].str.lower()

#removing the url
import re
def remove_urls(text):
    text = re.sub(r"http\S+", "",text)
    return text
df["review"] = df["review"].apply(remove_urls)    

#removing punctuations
def remove_punctuations(text):
    text = re.sub(r"^A-Za-b0-9\s", "",text)
    return text
df["review"]= df["review"].apply(remove_punctuations)

#removing html
def remove_html(text):
    text = re.sub(r"<.*?>", "",text)
    return text
df["review"] = df["review"].apply(remove_html) 

import nltk
#nltk.download("punkt") 
#nltk.download("punkt_tab") 
#nltk.download("stopwords")   

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
def remove_stopwords(text):
    tokens = word_tokenize(text)
    stop_words= stopwords.words("english")
    
    for word in tokens:
        if word in stop_words:
            text = text.replace(word , "")
    return text
df["review"] = df["review"].apply(remove_stopwords) 

        
    






