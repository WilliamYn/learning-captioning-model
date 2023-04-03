import re
import cv2
from collections import Counter
import pandas as pd
import numpy as np
import os
import json

# Open the file and read its data
def readFile (path):
    with open(path, encoding="utf8") as file:
        data = file.read()
    return data

# # Read captions from the file Flickr8k.token.txt
data = readFile ("data/textFiles/30k_captions.txt")

# # Split the data into each line, to get a list of captions
captions = data.split('\n')
# # Remove the last line since it is blank
captions = captions[:-1]

# Store the captions in a dictionary
# Each imageID will be mapped to a list of its captions

data_custom = readFile("data/textFiles/custom_captions.txt")
captions_custom = data_custom.split('\n')
for cap in captions_custom:
    if len(cap) != 0:
        captions.append(cap)

content = {}

for line in captions:
    imageID, caption = line.split('\t')

    parts = imageID.split("#")
    if len(parts) > 1 and "." not in parts[1]:
        imageID = parts[0]

    # If the imageID doesn't exist in the dictionary, create a blank entry
    if content.get(imageID) is None:
        content[imageID] = []

    # Append the current caption to the list of the corresponding image
    content[imageID].append(caption)

#Store the cleaned captions as a file "tokens_clean.json"

with open("data/textFiles/tokens_clean.json", "w") as f:
    json.dump(content, f)

print("Done")