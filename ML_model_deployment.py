#template inheritance
import re
from flask import Flask, redirect , url_for , render_template , request, session
from datetime import timedelta
from model import cleaner_trans
import pickle

app = Flask(__name__)
count_vectorizer = pickle.load(open("count_vectorizer.sav", 'rb'))
Encoder = pickle.load(open("Encoder.sav", 'rb'))
nb_classifier = pickle.load(open("model.pkl", 'rb'))
app =  Flask(__name__)
app.secret_key = "hello" #to encode and decode our data

nationalities = {
    "EG" : "Egyptian",
    "PL" : "Palestinian" ,
    "KW" : "Kuwaiti" ,    
    "LY" : "Libyan",
    "QA" : "Qatari",
    "JO" : "Jordanian",
    "LB" : "Lebanese",
    "SA" : "Saudi",
    "AE" : "Emirati"  ,
    "BH" : "Bahraini"   ,
    "OM" : "Omani",
    "SY" : "Syrian"    ,
    "DZ" : "Algerian"   ,
    "IQ" : "Iraqi"  ,
    "SD" : "Sudanese",   
    "MA" : "Moroccan"  , 
    "YE" : "Yemeni"    ,
    "TN" : "Tunisian"
}

@app.route('/prediction')
@app.route('/' , methods=["POST","GET"])
def predict():
    if request.method == "POST":
        text = request.form["phrase"]
        string_clean = cleaner_trans.fit_transform([text])
        vectorized_string = count_vectorizer.transform(string_clean)
        pred = nb_classifier.predict(vectorized_string)[0]
        text_to_displ = nationalities[Encoder.classes_[pred]]
        session["pred"] = text_to_displ  # create a session
        return redirect(url_for('result'))
    else:
        return render_template('index.html')

@app.route('/result', methods=["POST","GET"])
def result():
    if "pred" not in session:
        return redirect(url_for("predict"))
    else:
        if request.method == "POST":
            return redirect(url_for("predict"))
        text = session['pred']
        session.pop("pred" , None)
        return render_template("login.html",text=text)
if __name__ == "__main__":
    app.run(debug=True)
