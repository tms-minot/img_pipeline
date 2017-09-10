import mxnet as mx
import json
import csv
from skimage import io
import numpy as np

import rabbitmq as mq

def build_model():
    mod = mx.Symbol.load()

def infer():
    
    
def preprocess_img():
    
def load_img(url):
    try:
        img = io.imread(url)
    except:
        print "Invalid data or url"
    
    return img
        
def load_images(url_list)

    img_list = []
    for u in url_list:
        img_list.append({})
        img['url'] = u
        img['data'] = load_img(u)
    
    return img_list


de