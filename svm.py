import logging
import pandas as pd
import numpy as np
from numpy import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer,TfidfTransformer
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
df = pd.read_csv('input.txt', encoding="utf-8",error_bad_lines=False,sep=',')  #error_bad_lines=ignore 
df = df[pd.notnull(df['category'])]
#print(df.head(10))
#print(df['post'].apply(lambda x: len(x.split(' '))).sum())
#my_tags = ['1','2','3','4','5','6','7','8','9','10','11','12']
my_tags = ['1','2','3','4','5','6','7','8','9','10','11','12']

'''plt.figure(figsize=(10,4))
df.tags.value_counts().plot(kind='bar');
def print_plot(index):
    example = df[df.index == index][['post', 'tags']].values[0]
    if len(example) > 0:
        print(example[0])
        print('Tag:', example[1])'''
X = df['textual_features']
y = df.category
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)

sgd = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
               ])
sgd.fit(X_train, y_train)
from sklearn.metrics import classification_report
y_pred = sgd.predict(X_test)

print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred,target_names=my_tags))

