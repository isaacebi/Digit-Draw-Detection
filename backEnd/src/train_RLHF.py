# %% Importing Libraries
import os
import torch
import pathlib
import sqlite3
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

import pandas as pd
import base64
from PIL import Image
import io
import numpy as np

# %% Pathing
CURR_FILE = pathlib.Path(__file__).resolve()
PROJECT_DIR = CURR_FILE.parents[1]
DATA_PATH = os.path.join(PROJECT_DIR, 'data')
TRAIN_PATH = os.path.join(DATA_PATH, 'train')
TEST_PATH = os.path.join(DATA_PATH, 'test')
MODELS_PATH = os.path.join(PROJECT_DIR, 'models')
RLHF_PATH = os.path.join(DATA_PATH, 'RLHF.db')

# intialization
device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% Helper Function
# check folder is exist, if not then create folder
def check_path(folderPaths:list):
    for folderPath in folderPaths:
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)


def get_data(sql_path, tableName):
    # connecting sql
    conn = sqlite3.connect(sql_path)

    # query to pandas on forecast table
    df = pd.read_sql_query(f"SELECT * FROM {tableName}", conn)

    return df


def calculate_accuracy(model_output, target):
    # get the prediction
    predictions = torch.max(model_output, 1)[1].data.squeeze()
    
    # get the accuracy
    accuracy = (predictions == target).sum().item()/float(target.size(0))
    return accuracy

def preprocess_base64(image):
    # Decode the base64-encoded image string
    image_string = base64.b64decode(image)
    
    # Open and preprocess the image
    image = Image.open(io.BytesIO(image_string))
    image = image.convert('1') # Convert to black and white
    image = image.resize((28,28)) # Resize the image

    # Convert the image to a PyTorch tensor and move it to the specified device
    image_torch = torch.tensor(np.float32(np.array(image))).to(device)

    return image_torch.view(-1, 1, 28,28)

def train_model(model, num_epoch, loss_function, optimizer, df):

    # model training 
    model.train()
    
    for epoch in range(num_epoch):
        
        epoch_loss = 0
        epoch_accuracy = 0

        # Preprocess image
        df['image'] = df['image'].apply(lambda x: preprocess_base64(x))

        images = df['image']
        labels = torch.tensor(np.int32(df['number'])).to(device)

        # foward pass
        output = model(images)

        # calculate loss
        loss = loss_function(output, labels)

        # releasing the cache
        optimizer.zero_grad()

        # backward pass
        loss.backward()

        # update model parameter
        optimizer.step()

        # accumulate the loss and accuracy for each epoch
        epoch_loss += loss.item()
        epoch_accuracy += calculate_accuracy(output, labels)

        print(f"Epoch: {epoch} - Loss: {epoch_loss} - Accuracy: {epoch_accuracy/(index+1)}")


def test_model(model, df):
    # intialization
    device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accuracy = 0

    model.eval()

    # Preprocess image
    df['image'] = df['image'].apply(lambda x: preprocess_base64(x))
    
    for index, row in df.iterrows():
        # Convert the image to a PyTorch tensor and move it to the specified device
        images = torch.tensor(np.float32(np.array(row['image']))).to(device)

        labels = torch.tensor(np.number(np.array(row['number']))).to(device)

        # foward pass
        output = model(images)

        # Accumulate accuracy
        accuracy += calculate_accuracy(output, labels)

    # Print test accuracy
    print(f"Test Accuracy: {accuracy / (index + 1)}")

# %% Creating Model
# Ideally, it would be better to create a dedicated file to create and test CNN module architecture, 
# however since this is a simple problem, this approach is more preferable.
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # First Convolutional Layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        
        # Second Convolutional Layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        
        # Fully Connected Layer for Classification
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        # Forward pass through the first convolutional layer
        x = self.conv1(x)
        
        # Forward pass through the second convolutional layer
        x = self.conv2(x)
        
        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)  # flattening the layer      
        
        # Forward pass through the fully connected layer for classification
        output = self.out(x)
        
        return output
    

def main():
    df = get_data(RLHF_PATH, 'train_digit')

    # Check if GPU is available, and set the device accordingly
    device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the CNN model and move it to the specified device (GPU or CPU)
    cnn_model = CNN().to(device)

    # Define the loss function (cross-entropy) - for classification problem
    loss_function = nn.CrossEntropyLoss()

    # Define the optimizer (Adam) for updating the model parameters during training
    optimizer = optim.Adam(cnn_model.parameters(), lr=0.005)

    # training model
    train_model(
        model=cnn_model,
        num_epoch=10,
        loss_function=nn.CrossEntropyLoss(),
        optimizer=optim.Adam(cnn_model.parameters(), lr=0.005),
        df=df
    )

    # test model
    test_model(
        model=cnn_model,
        df=df
    )

    # Model Saving 
    # check path
    check_path([
        MODELS_PATH
    ])

    # Model path & name
    CNN_MODEL = os.path.join(MODELS_PATH, 'cnn_model.pth')

    #save the model state
    torch.save(cnn_model.state_dict(), CNN_MODEL)

    # finish indicator
    print(f"Model saved in : {MODELS_PATH}")


# %%
if __name__ == '__main__':
    main()
