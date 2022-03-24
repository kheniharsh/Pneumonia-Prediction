import numpy as np
from PIL import Image, ImageOps
from flask import *
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)

model_ct =  keras.models.load_model('prediction_ct.h5')

@app.route('/')
def index():
    
    return render_template('index.html', val='')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        file = "https://storage.googleapis.com/kagglesdsdata/datasets/17810/23812/chest_xray/test/PNEUMONIA/person100_bacteria_475.jpeg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20220322%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20220322T224520Z&X-Goog-Expires=345599&X-Goog-SignedHeaders=host&X-Goog-Signature=48b502787c510360e852d14642f9cac1a373066a46060a534443f1b7845327cbc5abdd68cc2fc598929338cc35dcbdfb2a496478ae880e49d721c5ab8bf476dcaf7efdd8b82ec433b76df8a9beb21d47652ee257b1164e7c3529fc5d096a602a14bb851741d58a5ad69e2d7dbda632e95f4771f5e56838ed3b85c509a95fc320c1a456ff8c30053bf991aa4f49a3f8cbcecb5fb552f04205965aa88942a0be5bd1f80868ebe989cdb1053f2278b3f0c5a5320e7613fa7422b3dac45f6dfc9bdf11226862e7e59fb8f368ed9f6e64fd0d1cb47013c00e33166663c92d14e57e11d418b3f02022bb556e76ab183b2df39498a09b32c2dc536dcd724d2b933d77d6"
    
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
    
    if classes_x[0]==0:
        render_template('home.html', val='Your Report is Normal')
    else:
        render_template('home.html', val='Oops. Report Consist Pneumonia, Consult a Doctor')

if __name__ == '__main__':
    app.run(port=8000)
