from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from ImageCaptioning import ImageCaption


app = Flask(__name__)

AIModel = None

@app.route('/')
def home():
	return render_template('home.html')  

@app.route('/predict',methods=['POST'])
def predict():
	global AIModel

	if AIModel is None: 
		AIModel = ImageCaption()

	file = request.files['image']

	file.save('./test.png')

	caption = AIModel.predict_captions('./test.png')

	return render_template('result.html',caption = caption)



if __name__ == '__main__':
	app.run(debug=True)