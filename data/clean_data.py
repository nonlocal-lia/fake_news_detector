# Recreates cleaning process from EDA notebook

print('Loading imports and functions')
import pandas as pd
import re
import sys  
sys.path.insert(0, '../code/')
import contraction
from nltk.corpus import stopwords

def clean(article):
    """
    Removing Reuters tags, special symbols, expands contractions, replaces hyperlinks with sting 'url',
    removes punctutation, lowercases text and removes doubled spacing
    """
    article=str(article)
    article=article.replace('(Reuters)','')
    article=article.replace('(reuters)','')
    
    # Replacing newline and tabs with space
    article=article.replace('\n',' ')
    article=article.replace('\t',' ')
            
    # Remove Special characters
    article = re.sub(r"\x89Û_", "", article)
    article = re.sub(r"\x89ÛÒ", "", article)
    article = re.sub(r"\x89ÛÓ", "", article)
    article = re.sub(r"\x89ÛÏ", "", article)
    article = re.sub(r"\x89Û÷", "", article)
    article = re.sub(r"\x89Ûª", "", article)
    article = re.sub(r"\x89Û\x9d", "", article)
    article = re.sub(r"\x89Û¢", "", article)
    article = re.sub(r"\x89Û¢åÊ", "", article)
    article = re.sub(r"&quot;", "", article)
        
    # Replacing URLs with url
    article = re.sub(r'http\S+', "url", article)  
    
    # Expand Contractions
    article = contraction.expand_contractions(article)
    
    # Remove Punctuation
    article = re.sub(r'[^\w\s]', ' ', article)
    
    # Lowercase for reduced Dimensionality
    article = article.lower()
    
    # Replace any created doublespaces
    article = re.sub("\s\s+", " ", article)
    
    return article

def remove_stopwords(article):
    """
    Given an string, returns a string with stop words removed
    """
    cached_stop_words = stopwords.words('english')
    return ' '.join([word for word in article.split() if word not in cached_stop_words])

print('Loading Data')
data = pd.read_csv('./complete_data.zip')
print('Data Loaded')
print('Cleaning Data')
data['totalwords'] = data['text'].str.split().str.len()
data.drop(data[data['totalwords'] < 10].index, inplace=True)
data['cleaned_text'] = data['text'].apply(clean)
data['cleaned_title'] = data['title'].apply(clean)
data['no_stop_text'] = data['cleaned_text'].apply(remove_stopwords)
data.drop(columns=['title', 'text', 'totalwords'], inplace=True)
compression_opts = dict(method='zip', archive_name='clean_data.csv')  
data.to_csv('./clean_data.zip', index=False, compression=compression_opts)
print('Data Saved')