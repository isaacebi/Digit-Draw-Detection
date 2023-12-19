# %% Importing Libraries
import os
import torch
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor


# %% Pathing
DIR_PATH = os.getcwd()
DATA_PATH = os.path.join(DIR_PATH, 'data')
TRAIN_PATH = os.path.join(DATA_PATH, 'train')
TEST_PATH = os.path.join(DATA_PATH, 'test')
RES_PATH = os.path.join(DIR_PATH, 'res')
MODELS_PATH = os.path.join(DIR_PATH, 'models')

# %% Helper Function
# check folder is exist, if not then create folder
def check_path(folderPaths:list):
    for folderPath in folderPaths:
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)

# to visualize data and save figure to specific folder (optional)
def visualize_data(dataset, savePath=None, save=False):
    # create figure size
    figure = plt.figure(figsize=(6, 6))
    
    # defined number of col & rows to be made
    cols, rows = 5, 5

    # gets random sample and plot it into 5 by 5 tile (plot 25 samples)
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(dataset), size=(1,)).item()
        img, label = dataset[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(label)
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")

    # save if true
    if save:
        plt.savefig(savePath)

    # shows plot
    plt.show()

def train_model(model, num_epoch, loss_function, optimizer, train_dataloader):
    """
    Function to train the CNN model.

    Parameters:
    - model: The DL model to be trained.
    - num_epoch: Iteration to be train
    - loss_function: The loss function used for training.
    - optimizer: The optimization algorithm used for updating model parameters.
    - train_dataloader: DataLoader providing training data.

    Returns:
    None
    """
    
    # model training 
    model.train()
    
    for epoch in range(num_epoch):
        
        epoch_loss = 0
        epoch_accuracy = 0
        i = 0
        for i, (images, labels) in enumerate(train_dataloader):

            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            output = model(images)
            
            # Calculate loss
            loss = loss_function(output, labels)
            
            # Releasing the cache
            optimizer.zero_grad() 
            
            # Backward Pass
            loss.backward()

            # Update model parameter
            optimizer.step()

            # accummulate the loss and accuracy for each epoch
            epoch_loss += loss.item()
            epoch_accuracy += calculate_accuracy(output, labels)

        print(f"Epoch: {epoch} - Loss: {epoch_loss} - Accuracy: {epoch_accuracy/(i+1)}")


def test_model(model, test_dataloader):
    """
    Function to test the CNN model on a given dataset.

    Parameters:
    - model: The trained DL model to be tested.
    - test_dataloader: DataLoader providing test data.

    Returns:
    None
    """

    model.eval()

    accuracy = 0
    i = 0
    for i, (images, labels) in enumerate(test_dataloader):

        images, labels = images.to(device), labels.to(device)

        # Forward pass
        output = model(images)

        # Accumulate accuracy
        accuracy += calculate_accuracy(output, labels)

    # Print test accuracy
    print(f"Test Accuracy: {accuracy / (i + 1)}")

# calculate model accuracy
def calculate_accuracy(model_output, target):
    # get the prediction
    predictions = torch.max(model_output, 1)[1].data.squeeze()
    
    # get the accuracy
    accuracy = (predictions == target).sum().item()/float(target.size(0))
    return accuracy

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





# %%
if __name__ == '__main__':
    # Data Loading
    # check path
    check_path([
        TRAIN_PATH, TEST_PATH,
    ])

    # download training & test data and put in DATA_PATH
    mnist_trainset = datasets.MNIST(
        root=TRAIN_PATH, 
        train=True, 
        download=True, 
        transform=ToTensor()
    )

    mnist_testset = datasets.MNIST(
        root=TEST_PATH, 
        train=False, 
        download=True, 
        transform=ToTensor()
    )


    #Data Visualization
    # check path
    check_path([
        RES_PATH
    ])

    # without save
    visualize_data(mnist_trainset)

    # create picture name
    toSavePicture = os.path.join(RES_PATH, 'sample.png')

    # with save
    visualize_data(mnist_trainset, toSavePicture, True)


    # Data Prep
    train_dataloader = DataLoader(
        mnist_trainset, 
        batch_size=20, 
        shuffle=True, 
        num_workers=0,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        mnist_testset, 
        batch_size=20, 
        shuffle=False, 
        num_workers=0,
        pin_memory=True
    )

    # Check if GPU is available, and set the device accordingly
    device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the CNN model and move it to the specified device (GPU or CPU)
    cnn_model = CNN().to(device)

    # Define the loss function (cross-entropy) - for classification problem
    loss_function = nn.CrossEntropyLoss()

    # Define the optimizer (Adam) for updating the model parameters during training
    optimizer = optim.Adam(cnn_model.parameters(), lr=0.005)

    # training model
    train_model(cnn_model, 10, loss_function, optimizer, train_dataloader)

    # test model
    test_model(cnn_model, test_dataloader)


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