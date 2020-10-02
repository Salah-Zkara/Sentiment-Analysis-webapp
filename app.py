from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import string
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import pickle

infile = open('classifier.pickle','rb')
NB_classifier = pickle.load(infile)
infile.close()
infile = open('unique_words.pickle','rb')
unique_words = pickle.load(infile)
infile.close()

def clear(sentence):
    language='english'
    sentence_clean = re.sub('@[A-Za-z0–9]+', '', sentence)
    sentence_clean = re.sub('https?:\/\/\S+', '', sentence_clean)
    sentence_clean = [char for char in sentence_clean if(char not in string.punctuation) ]
    sentence_clean = "".join(sentence_clean)
    sentence_clean = [word for word in sentence_clean.split() if(word.lower() not in stopwords.words(language)) ]
    return sentence_clean

def Tokenization(sentence):
    A=np.zeros(len(unique_words), dtype='uint8')
    L=[]
    for word in sentence:
        for i in range(len(unique_words)):
            if(word==unique_words[i]):
                A[i]=1
    L.append(A)
    return np.array((L),dtype='uint8')

app=Flask(__name__)

@app.route("/",methods=['GET','POST'])
def home():
    valid = "false"
    result=None
    if request.method=="POST":
        valid = "true"
        result = NB_classifier.predict(Tokenization( clear(request.form.get("myInpt")) ))[0]

    return render_template("index.html",valid=valid,message=result)

if __name__=="__main__":
    app.run(debug=True)