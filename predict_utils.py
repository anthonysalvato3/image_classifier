import tensorflow as tf
import numpy as np
from PIL import Image

def process_image(image_path):
    image = np.array(Image.open(image_path))
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image /= 255
    return image.numpy()