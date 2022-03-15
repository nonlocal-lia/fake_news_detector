# Data Collection Instructions

All needed data is in the repository, but if you want to recreate the data, use the following steps:

1) Clone https://github.com/KaiDMML/FakeNewsNet and follow the README instructions to scrap the articles. To make this quicker you can alter the ```data_features_to_collect``` in the  ```config.json``` in the code folder to only scrap ```news_articles``` this can avoid interacting with the twitter API or downloading a large amount of data, which was not used in this project.

2) Place the ```FakeNewsNet_articles_to_zip.py``` file in the code folder of the cloned repo and run it to collect the articles from the json files and save them as a zipped csv.

3) Download the data sets from https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset and https://www.kaggle.com/ruchi798/source-based-news-classification

4) Place the data files in the appropriate folder in this repo.

5) Run ```merge_data.py``` in this folder to recreate the complete dataset zip

6) Run ```clean_data.py``` or the EDA notebook to recreate the cleaned dataset zip
