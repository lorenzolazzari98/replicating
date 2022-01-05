# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 19:05:24 2022

@author: User
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.chdir("C:/Users/User/Documents/PR39/fasttext-commit-classification-master/data")

# INPUTTING LABELLED DATASET

labels=[]
messages=[]

with open("commits-labeled.txt", 'r', encoding="utf-8") as f:
    data=f.read()

import re

split=re.split('(__label__nonfunctional|__label__corrective|__label__unknown|__label__features|__label__perfective)', data)

for line in split:
    if line=='__label__nonfunctional' or line=='__label__corrective' or line=='__label__unknown' or line=='__label__features' or line=='__label__perfective':
        labels.append(line.replace("__label__",""))
        split.remove(line)

split.pop(0) #removing header

#check if last label matches with last element ----> OKK!
#check lenght len(labels), len (messages) -----> both 4528 OKK!!

## MINIMAL DATASET CLEANING

for line in split:
    ithmessage = line.rstrip('--\n')
    messages.append(ithmessage)
    
#CREATING DATAFRAME 

df=pd.DataFrame()
df['Labels']=labels
df['Messages']=messages

#PLOTTING LABELS COUNT

import pandas as pd
import seaborn as sns

sns.countplot(x="Labels", data=df,order = df['Labels'].value_counts().index,palette="crest")


#SPLITTING BETWEEN TRAINVAL AND TEST (as the paper test dataset does not take part on.., which number?? derived by confusione matrix 1127/4528 0 0.24, datset in the paper was larger so we do a 0.20)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df['Messages'], df['Labels'], test_size=0.2, random_state=0)

x_train.shape, y_train.shape


#FEATURE EXCRATACTION BY TF-IDF

from sklearn.feature_extraction.text import TfidfVectorizer

text_transformer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), lowercase=True, max_features=150000)
x_vect_train = text_transformer.fit_transform(x_train)
x_test_vect = text_transformer.transform(x_test) #check what does it do

### BASELINE MODELS DEFINITION

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, KFold, RepeatedKFold, StratifiedKFold


LT = LogisticRegression()
RF = RandomForestClassifier()
NB = MultinomialNB()
SVC = LinearSVC()
CART = DecisionTreeClassifier()

models_df=pd.DataFrame()
models_df['Models object']=[LT,RF,NB,SVC,CART]
models_df['Models reference']=['LT','RF','NB','SVC','CART']

##CV methods
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)
kf=KFold(n_splits=5, shuffle=True, random_state=77)
rkf = RepeatedKFold(n_splits=5, n_repeats=1, random_state=77)

##BASELINE MODELS EVALUATION

scores=[]

for model in models_df['Models object'].tolist() :
    cv_results = cross_val_score(model, x_vect_train, y_train, cv=kf, scoring='f1_weighted')
    scores.append(np.mean(cv_results))

models_df['F-measures']=scores
models_df.sort_values(by=['F-measures']).plot.barh(y='F-measures', x='Models reference')


#LOGIT PERFORMANCE ON INDEPENDENT TEST DATA

LT.fit(x_vect_train, y_train)
y_test_pred=LT.predict(x_test_vect)

from sklearn.metrics import f1_score, confusion_matrix, plot_confusion_matrix

f1_score(y_test_pred,y_test, average="weighted")
plot_confusion_matrix(LT, x_test_vect, y_test, cmap='Blues')




