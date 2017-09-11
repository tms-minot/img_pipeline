import os
import mxnet as mx
import json
import csv
from skimage import io
from scipy.misc import imresize
import numpy as np

model_path = os.path.realpath(os.path.dirname(__file__)+"/../model")
sym_path = model_path + '/Inception-7-symbol.json'
param_path = model_path + '/Inception-7-0001.params'
synset_path = model_path + '/synset.txt'
context = mx.cpu()
batch_size = 10
thr = 0.5


def build_module():
    ### This function builds the MXnet model of Inception V3 ###
    print '### Building MXNet module'
    sym = mx.sym.load(sym_path)                                             # load symbol    
    mod = mx.mod.Module(sym, context=context)                               # instantiate MXNet module
    mod.bind([("data",(batch_size,3,299,299))],
             [("softmax_label",(batch_size,))], 
             for_training=False)                                            # bind to memory, only for inference
    mod.load_params(param_path)                                             # load pretrained weights    
    return mod
    

def process_images(img_list, module):
    ### This function infers batches of data ###
    print '### Running inference'
    data = np.concatenate([im['data'] for im in img_list], axis=0)  
    labels = np.ones(data.shape[0])                                         # so that the engine doesn't complain
    eval_data = mx.io.NDArrayIter(data, labels, batch_size=batch_size)      # data iterator
    pred = module.predict(eval_data)                                        # predictions from softmax layer
    return pred                                                             # contains 8 junk classes, appended at the end

    
def preprocess_img(img):
    ### Preprocessing for Inception V3 ###

    h,w = img.shape[:2]
    crop = np.min([h,w])
    img = img[(h-crop)//2:(h+crop)//2, (w-crop)//2:(w+crop)//2, :]          # crop
    img = imresize(img, (299,299))                                          # resize for first layer
    img = img.astype('float32')
    img /= 255                                                      
    img -= 0.5
    img *= 2.
    img = img.transpose((2, 0, 1))                                          # channels first    
    img = np.expand_dims(img, axis=0)                                       # add dimension for batch
    return img
    
def make_results(img_list, pred):
    ### This function assembles the results from the predictions ###
    print '### Constructing output'
    syn = []
    with open(synset_path) as f:                                               # ImageNet synset names
        for line in f:
            syn.append(line[10:].split(',', 1)[0].split('\n',1)[0])         # first word in description 
            
    j = 0
    for im in img_list:
        if im['data'] is not None:
            im['classes'] = [{'class': syn[i], 'confidence':r.tolist()} for i,r in enumerate(np.nditer(pred[j])) if r>thr]
            j +=1
        else:
            im['Error'] = 'Invalid URL'
        im.pop('data', None)
    
    return img_list
    
def load_img(url):
    ### This function downloads an numpy image ###
    try:
        img = io.imread(url)
        assert len(img.shape) is 3
        assert img.shape[2] is 3
    except:
        print "### Invalid data or url"
        img = None
    
    return img
        
def load_imgs(url_list):
    ### This function builds a list of urls and its corresponding data ###
    
    img_list = []
    for u in url_list:
        print '### Loading image ' + u 
        img_list.append({})
        img_list[-1]['url'] = u
        img = load_img(u)
        img_list[-1]['data'] = preprocess_img(img) if img is not None else None
    
    return img_list

