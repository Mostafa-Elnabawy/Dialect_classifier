from model import cleaner_trans
import pickle

count_vectorizer = pickle.load(open("count_vectorizer.sav", 'rb'))
Encoder = pickle.load(open("Encoder.sav", 'rb'))
nb_classifier = pickle.load(open("model.pkl", 'rb'))
string_clean = cleaner_trans.fit_transform(["دخيلك معلم لا تقوصني"])
vectorized_string = count_vectorizer.transform(string_clean)
pred = nb_classifier.predict(vectorized_string)[0]
print("You are ",Encoder.classes_[pred])
