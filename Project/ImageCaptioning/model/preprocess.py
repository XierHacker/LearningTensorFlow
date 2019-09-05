import re
import numpy as np
import os
import time
import json
from glob import glob
from PIL import Image
import pickle
from sklearn.utils import shuffle

import model

annotation_file="/data3/xiekun/DataSets/coco/annotations/captions_train2014.json"
image_dir="/data3/xiekun/DataSets/coco/train2014/"

num_examples = 30000

def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path

def get_image_annotation():
    # Read the json file
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    # print("annotations:\n",annotations)

    # Store captions and image names in vectors
    all_captions = []
    all_img_names = []

    for annot in annotations['annotations']:
        caption = '<start> ' + annot['caption'] + ' <end>'
        image_id = annot['image_id']
        full_coco_image_path = image_dir + 'COCO_train2014_' + '%012d.jpg' % (image_id)

        all_img_names.append(full_coco_image_path)
        all_captions.append(caption)

    print("all_captions:\n", all_captions[0], len(all_captions))
    print("all_img_names\n", all_img_names[0], len(all_img_names))

    all_captions, all_img_names = shuffle(all_captions, all_img_names, random_state=1)
    print("all_captions:\n", all_captions[0], len(all_captions))
    print("all_img_names\n", all_img_names[0], len(all_img_names))

    train_captions = all_captions[:num_examples]
    train_img_names = all_img_names[:num_examples]
    return train_captions,train_img_names

if __name__=="__main__":
    train_captions, train_img_names=get_image_annotation()
    




