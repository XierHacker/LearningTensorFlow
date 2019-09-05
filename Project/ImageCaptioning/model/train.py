import re
import numpy as np
import os
import time
import json
from glob import glob
from PIL import Image
import pickle

import model

annotation_file="/data3/xiekun/DataSets/coco/annotations/captions_train2014.json"
image_path="/data3/xiekun/DataSets/coco/train2014/"


# Read the json file
with open(annotation_file, 'r') as f:
    annotations = json.load(f)

print("annotations:\n",annotations)
