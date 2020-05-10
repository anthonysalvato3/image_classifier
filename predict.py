import predict_utils as pu

import argparse
import numpy as np
import matplotlib.pyplot as plt
import json
import tensorflow as tf
import tensorflow_hub as hub

parser = argparse.ArgumentParser(description='Classify flowers.')
parser.add_argument('image_path', metavar='/path/to/image', help='The file path to the image')
parser.add_argument('model_path', metavar='/path/to/model', help='The file path to the model')
parser.add_argument('--top_k', metavar='K', type=int, help='Specify limiting the results to K')
parser.add_argument('--category_names', metavar='/path/to/json', help='A mapping of numerical categories to their names')
args = parser.parse_args()

image_path = args.image_path
model_path = args.model_path
top_k = args.top_k
category_names = args.category_names
class_names = None
if(category_names):
    with open(category_names, 'r') as f:
        class_names = json.load(f)
model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer':hub.KerasLayer})
model.summary()

im = np.expand_dims(pu.process_image(image_path), axis=0)
ps = model.predict(im)[0]
if top_k:
    probs = np.flip(ps[np.argsort(ps)[-top_k:]]).tolist()
else:
    probs = np.flip(ps[np.argsort(ps)]).tolist()
classes = []
for prob in probs:
    prob_index = np.where(ps == prob)
    class_label = np.asscalar(np.array(prob_index)) + 1
    if(not class_names):
        classes.append(class_label)
    else:
        classes.append(class_names[str(class_label)])
print(probs)
print(classes)
