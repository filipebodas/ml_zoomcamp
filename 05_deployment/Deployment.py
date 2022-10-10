#!/usr/bin/env python
# coding: utf-8

# ## Data preparation

# In[2]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# In[3]:


df = pd.read_csv("C:\\Users\\Geral\\Desktop\\VSCODE\\ml_zoomcamp\\03_classification\\Telco.csv")

df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)

df.churn = (df.churn == 'yes').astype(int)


# In[4]:


df_full_train, df_test = train_test_split(df, test_size = 0.2, random_state = 1)


# In[5]:


numerical = ['tenure', 'monthlycharges', 'totalcharges']

categorical = [
    'gender',
    'seniorcitizen',
    'partner',
    'dependents',
    'phoneservice',
    'multiplelines',
    'internetservice',
    'onlinesecurity',
    'onlinebackup',
    'deviceprotection',
    'techsupport',
    'streamingtv',
    'streamingmovies',
    'contract',
    'paperlessbilling',
    'paymentmethod',
]


# ### Train the model

# In[6]:


def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)
    
    return dv, model


# In[7]:


def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


# - Model parameters

# In[8]:


C = 1.0
n_splits = 5


# In[9]:


kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

scores = []

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.churn.values
    y_val = df_val.churn.values

    dv, model = train(df_train, y_train, C=C)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))


# In[10]:


scores


# In[11]:


dv, model = train(df_full_train, df_full_train.churn.values, C=1.0)
y_pred = predict(df_test, dv, model)

y_test = df_test.churn.values

auc = roc_auc_score(y_test, y_pred)
auc


# ## Saving and loading the model

# ### Save the model

# Library to save python objects

# In[12]:


import pickle


# Take the model and write it to a file
# 
# - Create a file where we will write the file

# In[13]:


output_file = f'model_C={C}.bin'
output_file


# Open functions opens a file and we specify what we want to do with it.
# 
# - In this case we want to "w"rite and the file will be "b"inary, we want to write bytes
# 
# - We need to write (dump function) both the model and the dictionary vectorizer using a tuple
# 
# - In the end we close the file

# In[14]:


f_out = open(output_file, 'wb')
pickle.dump((dv, model), f_out)
f_out.close()


# **Alternatively, we can use a with statement to ensure that the file is closed all the time**

# In[15]:


with open(output_file, "wb") as f_out:
    pickle.dump((dv, model), f_out)
    # do stuff
    
# do other stuff (file is closed)


# ### Load the model

# Restart the kernel to pretend we are in a new process, that doesn't know the previous variables
# 
# 
# Scikit-learn needs to be installed in the computer to create the DictVectorizer and the LogisticRegression classes

# In[1]:


model


# In[2]:


import pickle


# - Now we want to "r"ead th file
# 
# - Load to read from the file

# In[3]:


model_file = "model_C=1.0.bin"


# In[4]:


with open(model_file, "rb") as f_in:
    dv, model = pickle.load(f_in)


# In[5]:


dv, model


# - The DictVetorizer and the model (LR) are the variables that we saved previously

# Let's pretend we have a new customer with the following characteristics:

# In[6]:


customer = {
    "gender": "female",
    "seniorcitizen": 0,
    "partner": "yes",
    "dependents": "no",
    "phoneservice": "no",
    "multiplelines": "no_phone_service",
    "internetservice": "dsl",
    "onlinesecurity": "no",
    "deviceprotection": "no",
    "techsupport": "no",
    "streamingtv": "no",
    "streamingmovies": "no",
    "contract": "month-to-month",
    "paperlessbilling": "yes",
    "paymentmethod": "electronic_check",
    "tenure": 1,
    "monthlycharges": 29.85,
    "totalcharges": 29.85
}


# - Turn the customer data into a feature matrix

# In[7]:


X = dv.transform([customer])


# - Apply the model and obtain the probability for that customer to churn

# In[12]:


model.predict_proba(X)[0, 1]


# This is how a model can be saved and used later. However, it is not convinient to do this all in a jupyter notebook and run all the lines everytime the notebook is opened. There is a need to create a single file.

# - Download the Notebook as a .py file

# In[ ]:




