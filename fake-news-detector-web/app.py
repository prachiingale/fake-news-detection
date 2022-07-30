from flask import Flask, request, render_template
import pickle
import pandas as pd
import re
import string

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
vec = pickle.load(open('vec.pkl', 'rb'))

def output_lable(n):
    if n == 0:
        return "Fake "
    elif n == 1:
        return "Not Fake"
        
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text
   
def manual_testing(news):
    
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt) 
    new_x_test = new_def_test["text"]   
    new_xv_testt = vec.transform(new_x_test).toarray()
    pred_LR = model.predict(new_xv_testt)
    return output_lable(pred_LR[0])
    
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    val=request.form["article"]
    return render_template('index.html', prediction_text='Given News Article is {}'.format(manual_testing( str(val) )))

if __name__ == "__main__":
    app.run(debug=True)