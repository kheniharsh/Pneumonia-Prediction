import pickle
import joblib
import numpy as np
import cv2
from flask import Flask,render_template,request

from tensorflow import keras
model_ct = keras.models.load_model('prediction-ct.h5')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        predict_ct = []
        file = request.files['imagefile']
        img = cv2.imread(file.filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(gray,(150,150))
        predict_ct.append(img)
        predict_ct = np.array(predict_ct)
        predict_ct = predict_ct.reshape(predict_ct.shape[0], 150, 150,1)

        predict_x=model_ct.predict(predict_ct) 
        classes_x=np.argmax(predict_x, axis=1)
        classes_x
        if classes_x[0]==0:
            x='CT Scan predicted it's Normal'
            print(x)
        else:
            x='CT Scan predicted it's Pneumonia'
            print(x)
    return render_template('home.html', val=x)


if __name__ == '__main__':
    app.run()
