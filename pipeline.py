import mxnet as mx
import json
import csv
from skimage import io
from scipy.misc import imresize
import numpy as np

sym_path = 'model/Inception-7-symbol.json'
param_path = 'model/Inception-7-0001.params'
batch_size = 40
thr = 0.95


def build_module():
    ### This function builds the MXnet model of Inception V3 ###
    
    sym = mx.sym.load(sym_path)                                             # load symbol    
    mod = mx.mod.Module(sym, context=mx.cpu())                     # instantiate MXNet module
    mod.bind([("data",(batch_size,3,299,299))], for_training=False)   # bind to memory, only for inference
    mod.load_params(param_path)                                             # load pretrained weights    
    return mod
    

def process_images(img_list, module):
    ### This function infers batches of data ###
    
    data = np.stack([im['data'] for im in img_list])                        
    eval_data = mx.io.NDArrayIter(data, batch_size=batch_size)              # data iterator
    pred = module.predict(eval_data)                                        # predictions from softmax layer
    return pred

    
def preprocess_img(img):
    ### Preprocessing for Inception V3 ###
    
    h,w = img.shape[:2]
    
    crop = np.min([h,w])
    img = img.transpose((2, 0, 1))                                          # channels first
    img = img[:, (h-crop)//2:(h+crop)//2, (w-crop)//2:(w+crop)//2]          # crop
    img = imresize(img, (299,299))                                    # resize for first layer
    img = img.astype('float32')
    img /= 255                                                      
    img -= 0.5
    img *= 2.
    img = np.expand_dims(img, axis=0)                                       # add dimension for batch
    return img
    
def make_results(img_list, pred, thr):
    ### This function assembles the results from the predictions ###
    
    syn = []
    with open('model/synset.txt') as f:                                     # ImageNet synset names
        for line in f[1:]:
            syn.append(line[10:].split(' ', 1)[0])                          # first word in description 
            
    j = 0
    for im in img_list:
        if im['data'] is not None:
            im['classes'] = [{'class': syn[i], 'confidence':r} for i,r in enumerate(np.nditer(pred[j])) if r>thr]
            j +=1
        else:
            im['Error'] = 'Invalid URL'
        im.pop('data')
    
    return img_list
    
def load_img(url):
    ### This function downloads an numpy image ###
    try:
        img = io.imread(url)
        assert img.shape[2] is 3
    except:
        print "Invalid data or url"
    
    return img
        
def load_imgs(url_list):
    ### This function builds a list of urls and its corresponding data ###
    
    img_list = []
    for u in url_list:
        img_list.append({})
        img_list[-1]['url'] = u
        img_list[-1]['data'] = load_img(u)
    
    return img_list

