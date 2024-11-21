import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from PIL import Image
import os
import tensorflow as tf
import string
from pickle import dump
from pickle import load
from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import add
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.layers import LSTM, Embedding, Dropout
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def extract_features(image, model):
    try:
        image=Image.open(image)
        image = image.resize((299,299))
        image = np.array(image)
        if image.shape[2] == 4:
            image = image[..., :3]
        image = np.expand_dims(image, axis=0)
        image = image/127.5
        image = image - 1.0
        feature = model.predict(image)
        return feature
    except:
        print("ERROR: Can't open image! Ensure that image path and extension is correct")

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo, sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            print("None")
            break
        if word == 'end':
            break
        in_text += ' ' + word
    
    
    return in_text.split(' ', 1)[1]

tokenizer = load(open("tokenizer1.p","rb"))
img_path = "image.jpg"
img_path = "test.png"
img_path = "test_image.png"
img_path = "dog.jpg"
from keras.models import load_model

def predict(image):
    loaded_model = load_model('models/model_final.keras')
    xception_model = Xception(include_top=False, pooling="avg")
    photo = extract_features(image, xception_model)
    description = generate_desc(loaded_model, tokenizer, photo, 72)
    return description.split("while")[0].strip()
print(predict(img_path))