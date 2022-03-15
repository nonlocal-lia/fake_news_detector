# Should be placed in the code folder of the FakeNewsNet repo and run to make a zip file
# But only after the articles have been scraped using the the code in the repo
print('Importing packages')
import pandas as pd
import numpy as np
import os
import json
from os.path import exists

print('Loading Data')
fake_news_net = pd.DataFrame()
sources = ['politifact', 'gossipcop']
labels = ['fake', 'real']
for source in sources:
    for label in labels:
        for d in os.listdir('./fakenewsnet_dataset/'+ source +'/'+ label +'/'):
            if exists('./fakenewsnet_dataset/'+ source +'/'+ label +'/'+ d +'/news content.json'):
                with open('./fakenewsnet_dataset/'+ source +'/'+ label +'/'+ d +'/news content.json') as json_data:
                    data = json.load(json_data)
                df = pd.DataFrame([[data['title'], data['text'], label]])
                fake_news_net = pd.concat([fake_news_net, df], ignore_index=True)
print('Done Loading')
print('Cleaning Data')
fake_news_net.columns = ['title', 'text', 'label']
fake_news_net.replace("", np.nan, inplace=True)
fake_news_net.dropna(inplace=True)
print('Done Cleaning')
compression_opts = dict(method='zip',
                        archive_name='fake_news_net.csv')  
fake_news_net.to_csv('fake_news_net.zip', index=False,
          compression=compression_opts)
print('Data Saved as zip file')