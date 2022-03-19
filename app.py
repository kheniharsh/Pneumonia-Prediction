import numpy as np
from PIL import Image, ImageOps
from flask import Flask,render_template,request
from keras import models

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

model_ct = models.load_model('prediction-ct.h5')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['imagefile']
    
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
        x='Your Report is Normal'
    else:
        x='Oops. Report Consist Pneumonia, Consult a Doctor'

    return render_template('home.html', val=x)


if __name__ == '__main__':
    app.run()
