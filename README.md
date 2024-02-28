# Digit Draw Detection

## Overview

Introducing **Digit Draw Detection**, an interactive web application that empowers users to write any number from 0-9.

### Goals

- **Intelligent Analysis:** Leveraging deep learning CNN model to analyzes drawn content to extract the number drawn.

## Technologies Used

- **Backend:** [Python, PyTorch, Numpy, Flask]
- **Frontend:** [React]

### Prerequisites

Laptop with supported CUDA gpu, healthy body and lastly sanity to code full stack. Ah you also probably need to at least know how to google. <-- this skill help me a lot

## Getting Started

### 1. Clone the Repository

```
git clone https://github.com/isaacebi/Digit-Draw-Detection
```
or alternatively, you can just use GitHub Desktop like me :laughing:

### 2. Install Requirement

### 2.1 Front End Requirement

```
cd <frontEnd-file-located>
```

Next, in my case, open code editor (VScode) and run terminal the following command:

```
npm install
```

Incase you faced any issues, you can try google for it or just ask ChatGPT since I'm not very familiar with React. However, the following command was suggested by ChatGPT:

```
npm install --legacy-peer-deps
```

If everything is properly installed, just run the command:

```
npm run dev
```

then you should be good to go. By the way, you need node js to run npm so... yeah just download it at [node.js](https://nodejs.org/en)

### 2.2 Back End Requirement

Open a new cmd or anaconda and go the back end directory

```
cd <backEnd-file-located>
```

if in anaconda you can use:

```
# Create a Conda environment
conda create --name your-env-name python=3.8

# Activate the Conda environment
conda activate your-env-name
```

and inside the virtual environment:

```
conda install --file requirements.txt
```

if you are not using anaconda, then goodluck, I wont tell you how though.

## Usage

To run front end, just open the vscode and use:

```
npm run dev
```

While to run back end, just use:
```
python serve_model.py
```

both of the script are very depended on directory. So be sure to be in the right directory.

## Backend

Just building model using PyTorch, it take long time but yeah that how life work.

## Frontend

Literaly React. :neutral_face:

## Problem Solved

Upon investigation, the initial preprocess or version one preprocess is not suitable, the following can be shown below:


## Future Development

So now its the beginning of the year! Lets start with improving this model, right now the AI are simply producing guessing outcome. But why? Based on the coding, it might be a model drift. So lets make a check list for that:

- [ ] Making sure train data and real data are the similiar
- [ ] Need to make save data if model predict it wrong

Well, this section might be edited later after doing some tweak. For now, this is all for it.


## Acknowledgments

Huge shoutout to Krenovator for giving the emotional support needed for making this done in 1 hour.

## Contact

Feel free to anyone who can help you :laughing:

