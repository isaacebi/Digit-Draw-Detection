import base64
from PIL import Image
import io

def version_one(image):
    # Decode the base64-encoded image string
    image_string = base64.b64decode(image)
    
    # Open and preprocess the image
    image = Image.open(io.BytesIO(image_string))
    image = image.convert('1') # Convert to black and white
    image = image.resize((28,28)) # Resize the image
    return image


def version_two(image):
    # Decode the base64-encoded image string
    image_string = base64.b64decode(image)
    # Open and preprocess the image
    image = Image.open(io.BytesIO(image_string))
    image = image.resize((28,28)) # Resize the image
    image = image.convert('1') # Convert to black and white
    return image