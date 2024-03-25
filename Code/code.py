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

df2 = df.drop('text',axis=1)
df2.corr()
sns.heatmap(df2.corr(),annot=True)

DATA PREPROCESSING

from nltk.corpus import stopwords
nltk.download('stopwords')
# stopwords.words("english")

import string
string.punctuation

from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
# ps.stem('loving')

# lower case
# tokenization
# removing special characters
# removing stop words and punctuation
# stemming


def transform_text(text):
      text=text.lower()
      text=nltk.word_tokenize(text)
      y=[]
      for i in text :
        if i.isalnum():
          y.append(i)
      text = y[:]
      y.clear()

      for i in text:
        if i not in stopwords.words("english")and i not in string.punctuation:
          y.append(i)
      text =y[:]
      y.clear()

      for i in text:
          y.append(ps.stem(i))

      return " ".join(y)

df['transformed_text']=df['text'].apply(transform_text)
df.head()
from wordcloud import WordCloud
# Create a WordCloud object with desired parameters
wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')

spam_wc=wc.generate(df[df['target']==1]['transformed_text'].str.cat(sep=''))

plt.figure(figsize=(12,6))
plt.imshow(spam_wc)
ham_wc=wc.generate(df[df['target']==0]['transformed_text'].str.cat(sep=''))

plt.figure(figsize=(12,6))
plt.imshow(ham_wc)
spam_cor=[]
for msg in df[df['target']==1]['transformed_text'].tolist():
  for word in msg.split():
    spam_cor.append(word)

len(spam_cor)

from collections import Counter
counter_obj = Counter(spam_cor)
most_common_items = counter_obj.most_common(30)

df_most_common = pd.DataFrame(most_common_items, columns=['0', '1'])

sns.barplot(data=df_most_common, x='0', y='1')

plt.xticks(rotation='vertical')
plt.show()

ham_cor=[]
for msg in df[df['target']==0]['transformed_text'].tolist():
  for word in msg.split():
    ham_cor.append(word)


len(ham_cor)
counter_obj = Counter(ham_cor)
most_common_items = counter_obj.most_common(30)


df_most_common = pd.DataFrame(most_common_items, columns=['0', '1'])

sns.barplot(data=df_most_common, x='0', y='1')

plt.xticks(rotation='vertical')
plt.show()



