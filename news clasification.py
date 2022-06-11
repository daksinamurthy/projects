#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install nltk


# In[ ]:


import nltk


# In[7]:


nltk.download('punkt')


# In[2]:


import pandas as pd


# In[3]:


fake = pd.read_csv('Fake.csv')
genuine = pd.read_csv('True.csv')


# In[4]:


display(fake.info())
display(genuine.info())


# In[5]:


display(fake.head())
display(genuine.head())


# In[6]:


display(fake.subject.value_counts())
print('\n')
display(genuine.subject.value_counts())


# In[7]:


fake['target'] = 0
genuine['target'] = 1


# In[10]:


display(fake.head())


# In[11]:


display(genuine.head())


# In[12]:


data = pd.concat([fake, genuine], axis=0)


# In[13]:


data = data.reset_index(drop=True)


# In[14]:


data=data.drop(['subject','date','title'], axis=1)


# In[15]:


print(data.columns)
print(data)


# ##  TOKENIZATION

# In[16]:


from nltk.tokenize import word_tokenize
data['text']=data['text'].apply(word_tokenize)


# In[17]:


print(data.head(10))


# ## STEMMING 

# In[18]:


from nltk.stem.snowball import SnowballStemmer
porter = SnowballStemmer("english", ignore_stopwords=False)


# In[19]:


def stem_it(text):
    return [porter.stem(word) for word in text]


# In[20]:


data['text']=data['text'].apply(stem_it)


# In[21]:


print(data.head(10))


# ## Stopword removal

# In[22]:


#import nltk
#from nltk.corpus import stopwords
#nltk.download('stopwords')
#print(stopwords.words('english'))


# In[23]:


def stop_it(t):
    dt = [word for word in t if len(word)>2] 
    return dt


# In[24]:


data['text']=data['text'].apply(stop_it)


# In[25]:


print(data['text'].head(10))


# In[26]:


data['text']=data['text'].apply(' '.join)


# ## Splitting up of data 

# In[27]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['target'], test_size=0.25)
display(X_train.head())
print('\n')
display(y_train.head())


# ## Vectorization

# In[28]:


from sklearn.feature_extraction.text import TfidfVectorizer
my_tfidf = TfidfVectorizer( max_df=0.7)

tfidf_train = my_tfidf.fit_transform(X_train)
tfidf_test = my_tfidf.transform(X_test)


# In[29]:


print(tfidf_train)


# ## LogisticRegression

# In[30]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[31]:


model_1 = LogisticRegression(max_iter=900)
model_1.fit(tfidf_train, y_train)
pred_1 = model_1.predict(tfidf_test)
cr1    = accuracy_score(y_test,pred_1)
print(cr1*100)


# ## PassiveAggressiveClassifier

# In[32]:


from sklearn.linear_model import PassiveAggressiveClassifier

model = PassiveAggressiveClassifier(max_iter=50)
model.fit(tfidf_train, y_train)


# In[33]:


y_pred = model.predict(tfidf_test)
accscore = accuracy_score(y_test, y_pred)
print('The accuracy of prediction is ',accscore*100)


# In[ ]:




