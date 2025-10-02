# preprocess.py

import numpy as np
from PIL import Image

def preprocess_image(image: Image.Image, target_size=(224, 224)):
    # Convert to RGB just in case
    image = image.convert('RGB')
    image = image.resize(target_size)
    img_array = np.array(image) / 255.0  # Normalize
    return img_array
