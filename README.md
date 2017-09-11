This project is an API for image classification.
It infers from images' urls sent by HTTP requests and returns the classes detected in JSON format.

# Description

API calls are done by POST requests and providing a JSON object of urls.
The request is sent to a Flask web app which un turns redirects the taks to a Celery worker through a RabbitMQ queue.
This structure enables distributed computing across the Celery worker's threads. They share the MXNet inference engine, which is thread safe.
For all intents and purposes the deep learning model used within MXNet is Inception V3 pretrained on ImageNet, it spans a softmax output of 1000 classes.

* app.py is the Flask app
* pipeline.py contains the functions for image processing/inference

# Installation

## Dependencies
* MXNet
* Flask
* Celery
* RabbitMQ
* 
You can easily pip install all of them

Load the Inception model (cd to project dir):
    wget http://data.dmlc.ml/mxnet/models/imagenet/inception-v3.tar.gz
    tar -xfz inception-v3.tar.gz
    
## Up and running
The flask app and the Celery worker run in separate processes
Open a shell in the project directory and issue the following commands:
    sudo rabbitmq-server -detached
    celery -A app.py workers --loglevel=INFO --concurrency=2
    
In a second shell (project directory), run the Flask app:
    FLASK_APP=app.py flask run
    
Done!

# Usage
You can now send HTTP requests to the API, for example using the requests module in Python (see client sample script)

# NOTES
The paths and threshold variables are define in the pipeline.py script.
The API runs seemlessly on C9's CPU servers, although the tests were limited to 20 images and to localhost. Scaling it up shouldn't be an issue.
The number of Celery workers can be changed with the --concurrency flag.
The ports used are 5672 for RabbitMQ and 5000 for Flask.