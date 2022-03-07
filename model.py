import pandas as pd
import emoji
from sklearn.preprocessing import FunctionTransformer
import re
import string
from nltk.corpus import stopwords
import nltk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import pickle

def clean_data_text(text_col):
    # removing hashtags and mentions
    text_without_hashtags = [re.sub('[@#]\w+','',line) for line in text_col]
    print('Done removing hashtags')
    print('============================================')
    #removing english letters
    clear_text = [re.sub('[A-z]+|\d+','',line) for line in text_without_hashtags]
    print('Done removing foreign letters')
    print('============================================')
    # remove punctuation
    text_no_punct = list(map(lambda line : "".join([i for i in line if i not in string.punctuation]) , clear_text))
    print('Done removing punctuation')
    print('============================================')
    # no emojis
    emojis_iter = map(lambda y: y, emoji.UNICODE_EMOJI['en'].keys())
    regex_set = re.compile('|'.join(re.escape(em) for em in emojis_iter))
    text_no_emojis = [regex_set.sub('',line) for line in text_no_punct]
    print('Done removing emojis')
    print('============================================')
    #removing stop words
    nltk.download('stopwords')
    stopwords_list = stopwords.words('arabic')
    stopwords_deleted = list(map(lambda line : " ".join([word for word in line.split() if word not in stopwords_list]) , text_no_emojis))
    print('Done removing stopping words')
    print('============================================')
#     #stemming
#     stemmed = list(map(lambda line : " ".join([st.stem(word) for word in line.split()]) , stopwords_deleted))
#     print('Done ALL')
    return stopwords_deleted


cleaner_trans = FunctionTransformer(clean_data_text)
if __name__ == "__main__" :
    data = pd.read_csv('complete_data.csv')
    data.drop_duplicates(inplace=True)
    data.dropna(inplace=True)
    data_cleaned = data[data.text.apply(lambda line : type(line) == str)]
    data_cleaned['text'] = cleaner_trans.fit_transform(data_cleaned['text'])

    # Naive bayes model training
    X_train , X_test , y_train , y_test = train_test_split(data_cleaned['text'],data_cleaned['dialect'],test_size=0.3,random_state=42)
    count_vectorizer = CountVectorizer()
    count_train = count_vectorizer.fit_transform(X_train)
    pickle.dump(count_vectorizer, open("count_vectorizer.sav", 'wb'))
    Encoder = LabelEncoder()
    y_train_labeled = Encoder.fit_transform(y_train)
    pickle.dump(Encoder, open("Encoder.sav", 'wb'))
    nb_classifier = MultinomialNB()
    nb_classifier.fit(count_train, y_train_labeled)
    pickle.dump(nb_classifier, open("model.pkl", 'wb'))
    pred = nb_classifier.predict(count_train)
    score = metrics.accuracy_score(y_train_labeled ,pred )
    print("train score " , score)
    #Evaluating on Test Data
    count_test = count_vectorizer.transform(X_test)
    y_test_labeled = Encoder.transform(y_test)
    pred = nb_classifier.predict(count_test)
    score = metrics.accuracy_score(y_test_labeled ,pred )
    print("test score ", score)
