import mxnet as mx
import json
import csv
from skimage import io
import numpy as np

import rabbitmq as mq

def build_model():
    mod = mx.Symbol.load()

def infer():
    
    
def preprocess_img(img):
    h,w = img.shape[:2]
    img = img.astype('float32')
    crop = np.min(h,w)
    img = img.transpose((2, 0, 1)) # re-order dimensions
    img = img[:, (h-crop)//2:(h+crop)//2, (w-crop)//2:(w+crop)//2] #crop
    img = scipy.misc.imresize((299,299))
    img /= 255.
    img -= 0.5
    img *= 2.
    img = np.expand_dims(img, axis=0) # add dimension for batch
    return img
    
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