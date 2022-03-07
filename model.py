import pandas as pd
import emoji
from sklearn.preprocessing import FunctionTransformer
import re
import string
from nltk.corpus import stopwords
import nltk

data = pd.read_csv('complete_data.csv')
data.drop_duplicates(inplace=True)
data.dropna(inplace=True)
data_cleaned = data[data.text.apply(lambda line : type(line) == str)]

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
data_cleaned['text'] = cleaner_trans.fit_transform(data_cleaned['text'])

print(data_cleaned['text'].head())
