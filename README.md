__This project is an API for image classification.
It infers from images' urls sent by HTTP requests and returns the results in JSON format.__

# Description

The API calls are done by POST requests and providing a JSON object of urls.
Requests are sent to the Flask web app which in turns redirects the tasks to Celery workers through a RabbitMQ queue.

This structure enables distributed computing across the Celery worker's threads. They share the MXNet engine, which is thread safe for inference and has gpu capabilities.
For all intents and purposes the deep learning model used within MXNet is Inception V3 pretrained on ImageNet, it spans a softmax output of 1000 classes.

## Source code
* app.py is the Flask app
* pipeline.py contains the functions for image processing/inference
* client_sample.py provides a Python example of POST requests with JSON

## Unit tests
To do

## Model
Once extracted (see installation), it contains the JSON symbol, the pretrained parameters and the list of Imagenet synsets. 

# Installation

## Dependencies
* MXNet
* Flask
* Celery
* RabbitMQ

Pip installing all of the above is staightforward.

Load the Inception model (cd to project directory):
```
wget http://data.dmlc.ml/mxnet/models/imagenet/inception-v3.tar.gz
tar -xfz inception-v3.tar.gz
```

## Up and running
The flask app and the Celery worker run in separate processes.
Open a shell in the project directory and issue the following commands:
```
sudo rabbitmq-server -detached
celery -A src/app.py workers --loglevel=INFO --concurrency=2
```

In a second shell (project directory), run the Flask app:
```
FLASK_APP=src/app.py flask run
```    
Done!

# Usage
You can now post JSON payloads to the API on the route `/api/infer`, see example using the requests module in Python (client sample script):
```
POST localhost:5000/api/infer JSON={images:[url1, url2, url3]}
```

# NOTES
The paths and threshold variables are define in the pipeline.py script.
The API runs seemlessly on C9's CPU servers, although the tests were limited to 20 images and to localhost. Scaling up shouldn't be an issue.
The number of Celery workers can be changed with the `--concurrency` flag.
The ports used are 5672 for RabbitMQ and 5000 for Flask.