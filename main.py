import pickle

#import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


df = pd.read_json("dataset.json", lines= True)
df["label"] = df.annotation.apply(lambda x: x.get('label'))
df["label"] = df.label.apply(lambda x: x[0])

import nltk
#nltk.download(['punkt', 'wordnet'])

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

df["label"] = df.annotation.apply(lambda x: x.get('label'))
df["label"] = df.label.apply(lambda x: x[0])

df = df.drop(['annotation','extras'],axis='columns')



def load_data(path):
    df = pd.read_json(path, lines=True)

    df["label"] = df.annotation.apply(lambda x: x.get('label'))
    df["label"] = df.label.apply(lambda x: x[0])

    X = df.content.values
    y = df.label.values

    return X, y


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    # print(clean_tokens)
    return clean_tokens


def display_results(y_test, y_pred):
    labels = np.unique(y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred, labels=labels)
    accuracy = (y_pred == y_test).mean()


    print("Accuracy:", accuracy)


url = 'dataset.json'
X, y = load_data(url)
X_train, X_test, y_train, y_test = train_test_split(X, y)

    # initializing
vect = CountVectorizer(tokenizer=tokenize)
tfidf = TfidfTransformer()
clf = RandomForestClassifier()

    # train classifier
X_train_counts = vect.fit_transform(X_train)
X_train_tfidf = tfidf.fit_transform(X_train_counts)
clf.fit(X_train_tfidf, y_train)

pickle.dump(clf,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))


'''
    # predict on test data
X_test_counts = vect.transform(X_test)
X_test_tfidf = tfidf.transform(X_test_counts)
y_pred = clf.predict(X_test_tfidf)

    # predict on test data
X_test_counts = vect.transform(["whoa stop you stupid bitch sjw"])
comment = "whoa stop you stupid bitch sjw"
print("Comment given: " + comment)
print("Applying CountVectorizer on the given comment:")
print(X_test_counts)
X_test_tfidf = tfidf.transform(X_test_counts)
print("Applying Tfidf after CountVectorizer:")
print(X_test_tfidf)
print("Results after applying RandomForest Algorithm")
print("Given text: 'whoa stop you stupid bitch sjw' ")
print("Prediction: {}\n".format(clf.predict(X_test_tfidf)))
if (clf.predict(X_test_tfidf) == '1'):
    print("The comment that you had done now correlates to Bullying.Stop doing this!")
display_results(y_test, y_pred)
'''


