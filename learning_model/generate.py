import json
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import collections
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.models import Model, load_model
from PIL import Image
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

# Read the files word_to_idx.pkl and idx_to_word.pkl to get the mappings between word and index
word_to_index = {}
with open (current_dir + "/data/textFiles/word_to_idx.pkl", 'rb') as file:
    word_to_index = pd.read_pickle(file, compression=None)

index_to_word = {}
with open (current_dir + "/data/textFiles/idx_to_word.pkl", 'rb') as file:
    index_to_word = pd.read_pickle(file, compression=None)

max_len = 0
last_model = ""

with open (current_dir + "/data/textFiles/outputs.txt", "r") as file:
    for line in file:
        if line.startswith('max_len'):
            max_len = int(line.split('=')[1].strip())
        elif line.startswith('last_model'):
            last_model = line.split('=')[1].strip()

model = load_model(current_dir + last_model[1:])

resnet50_model = ResNet50 (weights = 'imagenet', input_shape = (224, 224, 3))
resnet50_model = Model (resnet50_model.input, resnet50_model.layers[-2].output)

def predict_caption(photo):
    """
    predict_caption generates captions for a random image using Greedy Search Algorithm
    :photo: The image to process
    return  The predicted caption
    """
    inp_text = "startseq"
    for _ in range(max_len):
        sequence = [word_to_index[w] for w in inp_text.split() if w in word_to_index]
        sequence = pad_sequences([sequence], maxlen=max_len, padding='post')
        ypred = model.predict([photo, sequence])
        ypred = ypred.argmax()
        word = index_to_word[ypred]
        inp_text += (' ' + word)
        if word == 'endseq':
            break

    final_caption = inp_text.split()[1:-1]
    final_caption = ' '.join(final_caption)
    return final_caption



def preprocess_image (img):
    """
    preprocess_image normalizes the image before encoding it
    Convert 3D tensor to a 4D tendor
    Normalize image accoring to ResNet50 requirement
    :img:   The image to caption
    return  The normalized image
    """
    target_size=(224, 224)
    img = img.resize(target_size)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


def encode_image (img):
    """
    encode_image is a wrapper function, which inputs an image and returns its encoding (feature vector)
    :img:   The image to caption
    return  A feature vector of the image
    """
    img = preprocess_image(img)

    feature_vector = resnet50_model.predict(img)
    return feature_vector

def runModel(img:Image):
    """
    runModel generates the caption of the image to return to the client
    The caption is generated from the last model trained.
    :img:         The image to caption
    return              A string of the image caption
    """
    photo = encode_image(img).reshape((1, 2048))

    caption = predict_caption(photo)

    print(caption)
    return caption
