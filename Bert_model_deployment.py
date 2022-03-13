#template inheritance
import re
from flask import Flask, redirect , url_for , render_template , request, session
from datetime import timedelta
from model import cleaner_trans
import pickle
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

tokenizer = BertTokenizer.from_pretrained("asafaya/bert-base-arabic")
app = Flask(__name__)
Encoder = pickle.load(open("Encoder.sav", 'rb'))
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
        string_tokenized = tokenizer(string_clean, padding=True, truncation=True, max_length=512)
        test_dataset = Dataset(string_tokenized)
        model = BertForSequenceClassification.from_pretrained("best_model", num_labels=18)
        test_trainer = Trainer(model)
        raw_pred, _, _ = test_trainer.predict(test_dataset)
        pred = np.argmax(raw_pred, axis=1)
        text_to_displ = nationalities[Encoder.classes_[pred[0]]]
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
