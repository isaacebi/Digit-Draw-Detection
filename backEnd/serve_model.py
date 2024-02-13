# %% Importing Libraries
import flask
from flask_cors import CORS
from flask import Flask, request, jsonify

import io
import os
import torch
import base64
import sqlite3
import numpy as np
import pandas as pd
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

from train_model import CNN

# %% Pathing
DIR_PATH = os.getcwd()
MODELS_PATH = os.path.join(DIR_PATH, 'models')
DATA_PATH = os.path.join(DIR_PATH, 'data')

# Model path & name
CNN_MODEL_PATH = os.path.join(MODELS_PATH, 'cnn_model.pth')

# DB
RLHF_DB = os.path.join(DATA_PATH, 'RLHF.db')

# %%
# Create a Flask application
app = Flask(__name__)

# Enable Cross-Origin Resource Sharing (CORS) for all routes
CORS(app, resources={r'/*' : {'origins': '*'}})

# Check if GPU is available, and set the device accordingly
device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the CNN model and move it to the specified device (GPU or CPU)
CNN_MODEL = CNN().to(device)

# Load pre-trained model weights
CNN_MODEL.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=device))
CNN_MODEL.eval()

# Define a route for handling predictions via POST requests
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == "POST":
        # Extract the image data from the POST request
        image = request.json['image']

        # Decode the base64-encoded image string
        image_string = base64.b64decode(image)
        
        # Open and preprocess the image
        image = Image.open(io.BytesIO(image_string))
        image = image.convert('1') # Convert to black and white
        image = image.resize((28,28)) # Resize the image

        # Convert the image to a PyTorch tensor and move it to the specified device
        image_torch = torch.tensor(np.float32(np.array(image))).to(device)

        # Perform inference using the CNN model
        output = CNN_MODEL(image_torch.view(-1, 1, 28,28))

	    # Apply softmax activation to get probabilities
        softmax_out = nn.Softmax(dim=1)(output)

        # Get the indices of the top two predictions
        top_pred = torch.topk(softmax_out[0], 2)[1]

        # Prepare the response in JSON format
        response = flask.jsonify({"predictions":top_pred.tolist()})
        
        return response

@app.route('/train', methods=['POST'])
def train():
    data = request.json  # Assuming JSON data is sent from the frontend
    # Process the received data here
    # print("Received data:", data)

    # extract picture
    image_str = request.json['image']
    
    # actual number - human feedback
    act_num = request.json['trainNumber']

    # Create a database connection and cursor using the 'with' statement
    with sqlite3.connect(RLHF_DB) as conn:
        cursor = conn.cursor()

        # Create the table if it doesn't exist
        create_sql = "CREATE TABLE IF NOT EXISTS train_digit (image TEXT, number INTEGER)"
        cursor.execute(create_sql)

        # Insert the received data into the database
        insert_sql = "INSERT INTO train_digit (image, number) VALUES (?, ?)"
        cursor.execute(insert_sql, (image_str, act_num))

        # Commit changes
        conn.commit()

    return jsonify({"message": "Data received successfully"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
