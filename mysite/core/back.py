import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from os import listdir
from os.path import isfile, join
from textblob import TextBlob
import string
from nltk.tokenize.treebank import TreebankWordDetokenizer
from langdetect import detect
import textract
import os


default_path = "/home/seemsred/Downloads/django-upload-example-master/media/books/"

upload_path = default_path + "pdfs/"

stop_words = set(stopwords.words('english'))

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))

my_file = os.path.join(THIS_FOLDER, 'text.csv')


def read(file):
    text = textract.process(file)
    return text.decode('utf-8')


def extract(cv):
    text = read(cv)
    text = str(text)
    text = text.replace("\n", " ")
    text = text.lower()
    return text


def parse():
    files = [join(upload_path, f) for f in listdir(upload_path) if isfile(join(upload_path, f))]
    i = 0
    temp_files = []
    # database = pd.DataFrame()
    skills = "mba office logistics english business analysis analytics purchase"
    while i < len(files):
        file = files[i]
        temp_files.append(files[i])
        dat = extract(file)
        lang = detect(dat)
        if lang == "ru":
            trans = TextBlob(dat)
            dat = trans.translate(from_lang='ru', to='en')
        i += 1
        data = str(dat)
        dataw = data.translate(str.maketrans('', '', string.punctuation))
        tokens = nltk.word_tokenize(dataw)
        words = [word for word in tokens if word.isalpha()]
        words = [w for w in words if not w in stop_words]
        words = TreebankWordDetokenizer().detokenize(words)
        # print(dat)
        write_file(str(i), skills, 0, words)
    return temp_files


def write_file(id, skills, result, resume):
    csv = open(my_file, "w+")
    columns = "id" + "," + "skills" + "," + "result" + "," + "resume"
    row = "\n" + id + "," + skills + "," + str(result) + "," + resume
    csv.write(columns)
    csv.write(row)


parse()

filename = os.path.join(THIS_FOLDER, 'finalized_model.sav')
# some time later...

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))

test_file = pd.read_csv(my_file, delimiter=",")

tagstest = test_file["result"].values.tolist()

featurestest = test_file["resume"].values.tolist()

test_features = []


#2
for sentence in range(0, len(featurestest)):
    # Remove all the special characters
    test_feature = re.sub(r'\W', ' ', str(featurestest[sentence]))

    # remove all single characters
    test_feature = re.sub(r'\s+[a-zA-Z]\s+', ' ', test_feature)

    # Remove single characters from the start
    test_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', test_feature)

    # Substituting multiple spaces with single space
    test_feature = re.sub(r'\s+', ' ', test_feature, flags=re.I)

    # Removing prefixed 'b'
    test_feature = re.sub(r'^b\s+', '', test_feature)

    # Converting to Lowercase
    test_feature = test_feature.lower()

    test_features.append(test_feature)

vectorizer1 = TfidfVectorizer(max_features=100, stop_words=stopwords.words('english'))
test_features = vectorizer1.fit_transform(test_features).toarray()
# print(test_features)
res = loaded_model.predict(test_features)
job = ""

if res == 1:
    job = "IT Manager"
elif res == 2:
    job = "Supply Manager"
elif res == 3:
    job = "Purchase Manager"
elif res == 4:
    job = "Not Acceptable"
# print(job)


text_file = os.path.join(THIS_FOLDER, 'result.txt')

text = open(text_file, "w+")
text.write(job)
