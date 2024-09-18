from main import tfidf,vect
from flask import Flask,render_template,request
import pickle
import numpy as np
import pandas as pd
app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))
@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/redir',methods=['POST'])
def redir():
    return render_template("https://mail.google.com/mail/u/0/#inbox?compose=new")
@app.route('/predict',methods=['POST','GET'])
def predict():

    comment = [(x) for x in request.form.values()]
    print(comment)
    final = [np.array(comment)]
    print(comment[0])
   # comment=comment[0]
    s=comment[0]

   # output = '{0:.{1}f}'.format(prediction[0][1], 2)
    #print(output)
    X_test_counts = vect.transform(comment)
    #comment = "whoa stop you stupid bitch sjw"
    #print("Comment given: " + comment)
    #print("Applying CountVectorizer on the given comment:")
    #print(X_test_counts)
    X_test_tfidf = tfidf.transform(X_test_counts)
    #print("Applying Tfidf after CountVectorizer:")
    #print(X_test_tfidf)
    print("Results after applying RandomForest Algorithm")
    #print("Prediction: {}\n".format(model.predict_proba(X_test_tfidf)))
    prediction = model.predict_proba(X_test_tfidf)
    output = '{0:.{1}f}'.format(prediction[0][1], 2)
    print(output)
    render_template('index.html', texta='{}'.format(s))
    #prediction = model.predict_proba(comment)
   # prediction = model.predict_proba(comment)
    #print(prediction)
    if output > str(0.4):
        return render_template('index.html', pred='Bullying Comment!!!\nProbability of being bully comment is {}. Report it!'.format(output),
                              )
    else:
        return render_template('index.html', pred='Not a Bullying Comment.\n Probability of being bully comment is {}'.format(output),
                               )


if __name__=='__main__':
    app.run()