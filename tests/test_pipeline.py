from os import sys
sys.path.insert(0, '../')
import pytest
import numpy
import pipeline

def test_img():
    img = pipeline.load_img('https://www.tensorflow.org/images/cropped_panda.jpg')
    assert type(img) is np.array
    img = pipeline.load('hello!')
    assert img is None
    
def test_preprocess():
    img = pipeline.preprocess_img(np.ones((10,10,3)))
    assert img.shape is (1,3,299,299)
    
def test_engine():
    mod = pipeline.build_module()
    assert type(mod) is MXnet

def test_classes():
    img_list = [{'url':'hey', 'data':'hi'}]
    pred = np.concatenate([np.ones(1,5), np.zeros(1,5)], axis=1)
    pipeline.make_results(img_list,pred)
    assert 'data' not in img_list[0].keys
    assert len(img_list[0]['classes']) is 5