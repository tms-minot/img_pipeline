import mxnet as mx
import json
import csv
from skimage import io
import numpy as np

import rabbitmq as mq

def build_module(batch_size):
    
    sym = mx.sym.load(sym_path)
    mod = mx.mod.Module(sym, context=gpu(0))
    mod.bind([("batch_data",(batch_size,3,299,299))], for_training=False)
    mod.load_params(param_path)
    return mod
    

def process_images(img_list, module):
    
    data = np.stack([im['data'] for im in img_list])
    eval_data = mx.io.NDArrayIter(data, batch_size=batch_size)
    pred = module.predict(eval_data)
    
    return pred

    
    
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
        
def load_images(url_list):

    img_list = []
    for u in url_list:
        img_list.append({})
        img_list[-1]['url'] = u
        img_list[-1]['data'] = load_img(u)
    
    return img_list

