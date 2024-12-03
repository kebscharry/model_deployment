import numpy as np
from flask import  request, jsonify, render_template, Flask
import pickle

from pyexpat import features

#create flask app
app = Flask(__name__)

#load pkl model
model = pickle.load(open('model.pkl', 'rb'))

#define home page
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])  #to receive ind var
def predict():
    # when I receive IV values, covert values to float and save to float features
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)

    return render_template('index.html', prediction_text='The flower species is {}'.format(prediction[0]))

if __name__ == '__main__':
    app.run(debug=True)