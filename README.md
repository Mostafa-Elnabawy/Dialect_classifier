# Dialect_classifier

## Description
*The project is to build a classification model to classify arabic dialects*

## Table of contents
- requirements.txt : file with all library used to create that project and run it locally  
- data_retrieval.py : Retrieving data script
- complete_data.csv : complete data retrieved
- Modeling:
   - Naive Bayes:
      - Encoder.sav : pickeled file for encoder used in training Naive Bayes model
      - NV_training.py : script used to preprocess data and train the Naive Bayes model
      - count_vectorizer.sav : pickled file for CountVectorizer used in training the Naive Bayes model
      - model.pkl : pickeled Naive Bayes model
   - Arab Bert:
      - best_model : arab bert base model
      - Bert_Encoder.sav : pickeled file for encoder used in training deep learning model
      - bert_training.ipynb : notebook ran on kaggle to train bert model
- Flask deployment:
   - Bert_model_deployment.py : script to run flask app with bert model
   - ML_model_deployment.py : script to run flask with Naive Bayes model
   - static : folder for css and images files used in flask web app 
   - templates : folder for html files


## How to run
1. on your terminal run pip install -r requirements.txt to install necessary libraries prefered in a virtual environment
2. run python3 Bert_model_deployment.py to open up a flask app on a local server where you can type the text and then hit predict to show your predicted dialec
3. or run python3 ML_model_deployment.py to run the flask app with the Naive Bayes model making the prediction

## References 
for more information about the arab bert model [Asafaya_HuggingFace](https://huggingface.co/asafaya/bert-base-arabic).
