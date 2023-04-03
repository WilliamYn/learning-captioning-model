# Image and data imports
import base64
from PIL import Image
import json
from io import BytesIO
import os

# Flask imports
from flask import Flask, request
from flask_cors import CORS
import logging

# Model imports
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from transformers import CLIPProcessor, CLIPModel
from deep_translator import GoogleTranslator
from learning_model import generate

# Azure storage imports
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

connect_str = 'DefaultEndpointsProtocol=https;AccountName=croppedfacesdataset;AccountKey=DTvc7Q8EQb0XBCUBiaWV/sWOnci1GbjfbMdhUhyzEFqL2EWDxtrZASnrkGfeL/QUyjiyrFF7b4/e+AStb3a86w==;EndpointSuffix=core.windows.net'
blob_service_client = BlobServiceClient.from_connection_string(connect_str)
container_name = 'imagecaptions'
captions_txt_blob_name = 'custom_captions.txt'
container_client = blob_service_client.get_container_client(container_name)
blob_client = blob_service_client.get_blob_client(container=container_name, blob=captions_txt_blob_name)

translator = GoogleTranslator(source='fr', target='en')

app = Flask(__name__)
CORS(app)

# Load captioning and zero shot models
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")


def get_tags_from_captions(captions: list) -> list:
    """
    Function that splits the 16 different captions into keywords to create the image tags
    :captions:          An array of the 15 captions generated with nucleus sampling and the caption generated with beam searrch
    return              An array of words (tags)
    """
    stop_words = set(stopwords.words('english'))
    tokenizer = RegexpTokenizer(r'\w+')
    tags = set()
    for caption in captions:
        words = tokenizer.tokenize(caption)
        for word in words:
            if word not in stop_words:
                tags.add(word)
    return list(tags)


def zero_shot_classification(raw_image, classes):
    """
    Takes an image and a list of words (the potential tags) and gives a score to each word describinging the
    percentage of the image occupied by that word by using a Zero Shot classification model
    :raw_image:         The image on which to apply zero shot classification
    :classes:           A list of tags
    return              Tuples with tags and the percentage of the tag in the image
    """
    # CLIP:
    if len(classes) == 0:
        return zip([], [])
    inputs = processor(text=classes, images=raw_image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1).cpu().detach().numpy()[0]
    return zip(classes, probs)


def get_azure_custom_captions_txt_file():
    return blob_client.download_blob().content_as_text()


def update_azure_custom_captions_txt_file(file_contents):
    blob_client.upload_blob(file_contents, overwrite=True)


def upload_single_caption_to_azure(image, index, file_type):
    azure_subdirectory_name = f"images/"
    filename = f"image_{index}.{file_type}"

    blob_name = os.path.join(azure_subdirectory_name, filename)
    blob_client = container_client.get_blob_client(blob=blob_name)

    try:
        blob_client.upload_blob(image, overwrite=True)
    except:
        return json.dump({"status": 'error'})


@app.route('/')
def hello():
    return 'Hello World! -Caption Service'


@app.route("/caption", methods=["POST"])
def handle_add_single_caption():
    data = request.get_json()
    image, caption, file_type = data["image"], data["caption"], data["fileType"]

    # Get txt file from azure
    file_contents = get_azure_custom_captions_txt_file()

    # Add new caption to the txt file
    index = len(file_contents.split("\n"))
    new_caption = f"image_{index}.{file_type}\t{translator.translate(caption)}\n"
    file_contents += new_caption

    # Update the txt file
    print('Update txt file on azure')
    update_azure_custom_captions_txt_file(file_contents)

    # Add image to azure blob
    print('Uploading image to azure')
    upload_single_caption_to_azure(image, index, file_type)

    return json.dumps({'status': 'success'})


@app.route("/learning-model", methods=["GET", "POST"])
def handle_request():
    """
    Handles the reception of the request from the client and returns a response
    """
    if request.method == "POST":
        # Extract image base64 from request body
        request_data = request.get_json()
        json_key = "image"
        if json_key not in request_data:
            return f"The JSON data must have {json_key} as a key with the base64 encoding", 400
        
        base64_str = request_data[json_key]
        utf8_encoding = base64_str.encode(encoding='utf-8')
        image_bytes_io = BytesIO(base64.b64decode(utf8_encoding))
        
        image_caption = ""              # The most accurate caption of the image
        tags_and_probabilities = []     # A list words (tags) taken from nucleus_sampling_captions with their respective percentage of presence in the image
        
        # Open image received in request
        op_image = None
        try:
            op_image = Image.open(image_bytes_io)
        except Exception as e:
            return f"Could not open image: {e}", 400

        # Apply AI models on image
        # try:
        # except Exception as e:
        #     return f"Could not generate metadata for this image: {e}", 400
        raw_image = op_image.convert("RGB")
        app.logger.debug("Raw image opened")

        image_caption = generate.runModel(raw_image)
        app.logger.debug("Caption generated")

        tags = get_tags_from_captions([image_caption])
        app.logger.debug("Vocab generated")

        tags_and_probabilities = zero_shot_classification(raw_image, tags)
        if tags_and_probabilities == zip([], []):
            tags_and_probabilities = zip(["no_tags_found"], [1])
        app.logger.debug("Image classified into classes")
        op_image.close()
        image_bytes_io.close()

        return json.dumps({"tags": [(tag, float(prob)) for tag, prob in tags_and_probabilities], "captions": [image_caption], "english_cap": [image_caption]}, separators=(',', ':')), 200
    else:
        return "This endpoint only accepts POST requests", 400


if __name__ == "__main__":
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    app.logger.addHandler(handler)
    app.logger.setLevel(logging.DEBUG)
    app.run(host="0.0.0.0", port=80)
