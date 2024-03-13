import numpy as np
import pandas as pd
df = pd.read_csv('spam.csv',encoding='ISO-8859-1')

#Data Cleaning
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
#renaming the columns
df.rename(columns={'v1':'target','v2':'text'},inplace=True)
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])

#check missing values
df.isnull().sum()
#check for duplicate values
df.duplicated().sum()
#remove duplicates
df = df.drop_duplicates(keep='first')

#EDA (Exploratory Data Analysis)
df['target'].value_counts()
#pie chart
import matplotlib.pyplot as plt
plt.pie(df['target'].value_counts(), labels=['ham','spam'], autopct="%0.2f")
plt.show()
#data is imbalanced
import nltk
nltk.download('punkt')

df['num_characters']= df['text'].apply(len)   #number of characters

df['num_words'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x))) #number of words

df['num_sentences'] = df['text'].apply(lambda x: len(nltk.sent_tokenize(x))) #number of sentences

df[['num_characters','num_words','num_sentences']].describe()

#ham
df[df['target'] == 0][['num_characters','num_words','num_sentences']].describe()

#spam
df[df['target'] == 1][['num_characters','num_words','num_sentences']].describe()

import seaborn as sns
sns.histplot(df[df['target']==0]['num_characters'])
sns.histplot(df[df['target']==1]['num_characters'], color='red')

sns.histplot(df[df['target']==0]['num_words'])
sns.histplot(df[df['target']==1]['num_words'], color='red')

sns.histplot(df[df['target']==0]['num_sentences'])
sns.histplot(df[df['target']==1]['num_sentences'], color='red')

sns.pairplot(df,hue='target')



