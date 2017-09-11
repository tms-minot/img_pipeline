import json
import requests

r = requests.get('http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n02084071')    # dog class
urls = r.content.split('\r\n',20)[:20]      
urls.append('https://www.tensorflow.org/images/cropped_panda.jpg')                              # making sure panda returns panda
payload = {'images': urls}

p = requests.post('http://localhost:5000/api/infer', json=payload)                              # send to flask