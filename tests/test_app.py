import requests
r = requests.get('http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n02084071')
urls = r.content.split('\r\n',80)[:80]
img_list = pipeline.load_imgs(urls[:30])
img_l = [im for im in img_list if im['data'] is not None]
mod = pipeline.build_module()
pred = pipeline.process_images(img_l, mod).asnumpy()

import json
import requests
r = requests.get('http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n02084071')
urls = r.content.split('\r\n',80)[:80]
payload = {'images': urls}

p = requests.post('http://localhost/api/infer', json=payload)