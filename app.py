import numpy as np
from PIL import Image, ImageOps
from flask import Flask,render_template,request
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)
model_ct =  keras.models.load_model('prediction_ct.h5')

@app.route('/')
def index():
    
    return render_template('index.html', val='')

@app.route('/predict',methods=['POST'])
def predict():
    file=request.files['imagefile']
    predict_ct = []    
    img = Image.open(file)
    gray = ImageOps.grayscale(img)
    img = gray.resize((150,150), Image.ANTIALIAS)
    imgs = np.asarray(img)
    predict_ct.append(imgs)
    predict_ct = np.array(predict_ct)
    predict_ct = predict_ct.reshape(predict_ct.shape[0], 150, 150,1)

    predict_x=model_ct.predict(predict_ct) 
    classes_x=np.argmax(predict_x, axis=1)
    classes_x
    if classes_x[0]==0:
        return render_template('index.html', val='Your Report is Normal')
    else:
        return render_template('index.html', val='Oops. Report Consist Pneumonia, Consult a Doctor')

if __name__ == '__main__':
    app.run()
