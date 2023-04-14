from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, LSTM, add
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from pickle import dump, load
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from time import time
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import json
import os
# Compute the frequency of occurrence of each word, and store it in a dictionary of word-freq
import collections
import numpy as np
# Store the above computed features on the disk
# Use pickle to dump the entire data
import pickle

content = None
images_path = "data/Images/images"

# Read the file tokens_clean.json and store the cleaned captions in a dictionary
with open ("data/textFiles/tokens_clean.json") as file:
    content = json.load(file)

#Iterate over the captions word by word, and append each word to total_words
total_words = []

for key in content.keys():
    for caption in content[key]:
        for i in caption.split():
            total_words.append(i)

counter = collections.Counter(total_words)
freq_cnt = dict(counter)

# Store the word-freq pairs (from the dictionary freq_cnt) in a list, sorted in decreasing order of frequency
sorted_freq_cnt = sorted(freq_cnt.items(), reverse=True, key=lambda x:x[1])

threshold = 10

#Filter off those words whose frequency of occurrence in less than threshold
sorted_freq_cnt = [x for x in sorted_freq_cnt if x[1]>threshold]
# Store these common words in total_words
total_words = [x[0] for x in sorted_freq_cnt]

# Read training and testing image names

train_file_data = ""
test_file_data = ""

with open ("data/textFiles/flickr30k_train.txt", 'r') as file:
    train_file_data = file.read()

with open ("data/textFiles/flickr30k_test.txt", 'r') as file:
    test_file_data = file.read()

# # Obtain a list of train and test images
train_data = [img_file_name for img_file_name in train_file_data.split("\n")[:-1]]
test_data = [img_file_name for img_file_name in test_file_data.split("\n")[:-1]]

with open ("data/textFiles/custom_captions.txt", 'r') as file:
    custom_file_data = file.read()

numberOfElements = len([file for file in custom_file_data.split("\n") if len(file) != 0])
counter = -1
for i in custom_file_data.split("\n"):
    counter += 1
    file_name = i.split("\t")[0]
    if len(file_name) == 0:
        continue
    if counter < 0.9 * numberOfElements:
        train_data.append(file_name)
    else:
        test_data.append(file_name)

train_data = [image for image in train_data if os.path.isfile(os.path.join(images_path, image))]
test_data = [image for image in test_data if os.path.isfile(os.path.join(images_path, image))]

if len(test_data) + len(train_data) <= 1:
    raise Exception(f"Not enough images in {images_path} folder")

if len(test_data) == 0:
    test_data.append(train_data[-1])

if len(train_data) == 0:
    train_data.append(test_data[-1])

# For each imageID in train_data, store its captions in a dictionary 
train_content = {}
for imageID in train_data:
    train_content[imageID] = []
    for caption in content[imageID]:
        # Add a start sequence token in the beginning and an end sequence token at the end
        cap_to_append = "startseq " + caption + " endseq"
        train_content[imageID].append(cap_to_append)

model = ResNet50(weights = 'imagenet', input_shape = (224, 224, 3))
model.summary()
model_new = Model (model.input, model.layers[-2].output)

def preprocess_image (img):
    img = image.load_img(img, target_size=(224, 224))
    img = image.img_to_array(img)

    # Convert 3D tensor to a 4D tendor
    img = np.expand_dims(img, axis=0)

    #Normalize image accoring to ResNet50 requirement
    img = preprocess_input(img)

    return img

# A wrapper function, which inputs an image and returns its encoding (feature vector)
def encode_image (img):
    img = preprocess_image(img)
    feature_vector = model_new.predict(img)
    feature_vector = feature_vector.reshape((-1,))
    return feature_vector



train_encoding = {}
# Create a dictionary of iamgeID and its feature vector
for index, imageID in enumerate (train_data):
    image_path = os.path.join(images_path, imageID)
    if not os.path.isfile(image_path):
        continue
    train_encoding[imageID] = encode_image(image_path)
    # Print progress
    if index%100 == 0:
        print("Encoding in progress... STEP", index)


with open("encoded_train_features.pkl", "wb") as file:
    # Pickle allows to store any object as a file on the disk
    pickle.dump(train_encoding, file)

test_encoding = {}
# Create a dictionary of iamgeID and its feature vector

for index, imageID in enumerate (test_data):
    image_path = os.path.join(images_path, imageID)
    if not os.path.isfile(image_path):
        continue
    test_encoding[imageID] = encode_image(image_path)
    # Print progress
    if index%100 == 0:
        print("Encoding in progress... STEP", index)

with open("encoded_test_features.pkl", "wb") as file:
    pickle.dump(test_encoding, file)

# Create the word-to-index and index-to-word mappings
word_to_index = {}
index_to_word = {}

for i, word in enumerate(total_words):
    word_to_index[word] = i+1
    index_to_word[i+1] = word

size = len(index_to_word)

# Add startseq and endseq also to the mappings
index_to_word[size] = 'startseq'
word_to_index['startseq'] = size

index_to_word[size + 1] = 'endseq'
word_to_index['endseq'] = size + 1

VOCAB_SIZE = len(word_to_index) + 1

print("vocabsize", VOCAB_SIZE)

with open("data/textFiles/word_to_idx.pkl", "wb") as file:
    pickle.dump(word_to_index, file)

with open("data/textFiles/idx_to_word.pkl", "wb") as file:
    pickle.dump(index_to_word, file)

# Get the maximum length of a caption
max_len = 0

for cap_list in train_content.keys():
    for caption in train_content[cap_list]:
        max_len = max(max_len, len(caption.split()))

# Get the Glove word Embeddings
# This contains 50-dimensional embeddings for 6 Billion English words
file = open("glove.6B.50d.txt",encoding='utf8')

# Create a mapping from word to embedding
embeddings_index = {} # empty dictionary

for line in file:
    values = line.split()

    word = values[0]
    coefs = np.array (values[1:], dtype='float')
    embeddings_index[word] = coefs

file.close()

embedding_dim = 50

embedding_matrix = np.zeros((VOCAB_SIZE, embedding_dim))

for word, i in word_to_index.items():
    #if i < max_words:
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in the embedding index will be all zeros
        embedding_matrix[i] = embedding_vector


#Convert feature vector of image to smaller vector
#Output of ResNet goes into following input layer 
inp_img_features = Input(shape=(2048,))

inp_img1 = Dropout(0.3)(inp_img_features)
inp_img2 = Dense(256, activation='relu')(inp_img1)

#Now take Captions as input
#Actual input size will be (batch_size x max_length_of_caption)
#But here we specify only for one example
inp_cap = Input(shape=(max_len,))
inp_cap1 = Embedding(input_dim=VOCAB_SIZE, output_dim=50, mask_zero=True)(inp_cap)
inp_cap2 = Dropout(0.3)(inp_cap1)
inp_cap3 = LSTM(256)(inp_cap2)
# inp_cap3 captures the entire sentence that has been generated till now

# Decode the inputs
# So, an image (224x224x3) goes through ResNet50
# Then as 2048 dimensional it goes through the above earlier architecture
# The final output is inp_img2 (256 dimensional) which now goes through the Decoder 

# Similarly for the captions which initially have shape (batch_size x max_len)
# Then after passing through Embedding layer comes out as (batch_size x max_len x 50(embedding_size)))
# Then it passes through the above LSTM layer and comes out as inp_cap3 (a 256 dimensional vector)

# Add the two above tensors
decoder1 = add([inp_img2, inp_cap3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(VOCAB_SIZE, activation='softmax')(decoder2)

# Combined model
model = Model (inputs=[inp_img_features, inp_cap], outputs=outputs)
model.summary()

model.layers[2].set_weights([embedding_matrix])
model.layers[2].trainable = False
model.compile(loss="categorical_crossentropy", optimizer="adam")

def data_generator (train_content, train_encoding, word_to_index, max_len, batch_size):
    X1, X2, y = [], [], []
    n = 0

    while True:
        for imageID, cap_list in train_content.items():
            n += 1

            image = train_encoding[imageID]

            for caption in cap_list:
                idx_seq = [word_to_index[word] for word in caption.split() if word in word_to_index]

                for i in range (1, len(idx_seq)):
                    xi = idx_seq[0 : i] # The input sequence of words
                    yi = idx_seq[i] # The next word after the above sequence (this is expected to be predicted)

                    # Add a padding of zeros ao lengths of input sequences become equal
                    xi = pad_sequences([xi], maxlen=max_len, value=0, padding='post')[0] # Take the first row only, since this method inputs & returns a 2D array
                    # Convert the expected word to One Hot vector notation
                    yi = to_categorical([yi], num_classes=VOCAB_SIZE)[0]

                    X1.append(image)
                    X2.append(xi)
                    y.append(yi)

                if n==batch_size:
                    yield [[np.array(X1), np.array(X2)], np.array(y)]
                    
                    X1, X2, y = [], [], []
                    n=0

epochs = 10
batch_size = 5

with open ("data/textFiles/inputs.txt", "r") as file:
    for line in file:
        if line.startswith('epochs'):
            try:
                epochs = int(line.split('=')[1].strip())
            except ValueError:
                print("Error: could not convert epochs to an integer. Using default value of 10.")
        elif line.startswith('batch_size'):
            try:
                batch_size = int(line.split('=')[1].strip())
            except ValueError:
                print("Error: could not convert batch size to an integer. Using default value of 5.")

if epochs < 0:
    epochs = 1

if batch_size < 0:
    batch_size = 1
elif batch_size > len(train_content):
    batch_size = len(train_content)

steps = len(train_content)//batch_size

last_model = ""

for i in range(epochs):
    # Create an instance of the generator
    generator = data_generator(train_content, train_encoding, word_to_index, max_len, batch_size)
    model.fit(generator, steps_per_epoch=steps)
    model.save('./model_checkpoints/model_' + str(i) + '.h5')

last_model = f'./model_checkpoints/model_{epochs - 1}.h5'

with open("data/textFiles/outputs.txt", "w+") as file:
    file.write(f"max_len={max_len}\n")
    file.write(f"last_model={last_model}\n")


print("Done")
