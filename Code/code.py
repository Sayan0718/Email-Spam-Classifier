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
