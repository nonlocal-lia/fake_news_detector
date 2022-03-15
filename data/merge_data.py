# Run to recreate the merged complete dataset from the three sources

print('Importing Packages')
import pandas as pd

print('Loading Data')
fake_news_net = pd.read_csv('./FakeNewsNet/fake_news_net.zip')
fake = pd.read_csv('./Fake_and_real_news_dataset/Fake.csv')
real = pd.read_csv('./Fake_and_real_news_dataset/True.csv')
news = pd.read_csv('./Source_Based_Fake_News_Classification/news_articles.csv')
print('Data Loaded')
print('Cleaning Data for Merging')
fake['label']='fake'
fake.drop(columns=['subject','date'], inplace=True)
real['label']='real'
real.drop(columns=['subject','date'], inplace=True)
news_filtered = news[['title', 'text', 'label']]
news_filtered['label'] = news_filtered['label'].map(lambda x: str(x).lower())
news_filtered.dropna(inplace=True)
print('Data Cleaned')
print('Merging Data')
complete = pd.concat([fake_news_net, fake, real, news_filtered], ignore_index=True)
compression_opts = dict(method='zip',
                        archive_name='complete_data.csv')  
complete.to_csv('./data/complete_data.zip', index=False,
          compression=compression_opts)
print('Data Saved')