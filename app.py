#!flask/bin/python
from flask import Flask, request, jsonify
import pipeline
from celery import Celery

def make_celery(app):
    celery = Celery(app.import_name, backend=app.config['CELERY_RESULT_BACKEND'],
                    broker=app.config['CELERY_BROKER_URL'])
    celery.conf.update(app.config)
    TaskBase = celery.Task
    class ContextTask(TaskBase):
        abstract = True
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return TaskBase.__call__(self, *args, **kwargs)
    celery.Task = ContextTask
    return celery
    
app = Flask(__name__)                                                       # http app with rabbitmq broker
app.config.update(
    CELERY_BROKER_URL='amqp://guest:guest@localhost:5672/',
    CELERY_RESULT_BACKEND='amqp://guest:guest@localhost:5672/')
celery = make_celery(app)                                                   # async job queue for workers to process


@app.route('/api/infer', methods=['POST'])                                  # API entry point decorator
def post_data():
    payload = request.get_json(force=True)                                                # JSON input
    results = worker_process.apply_async((payload,))
    resp = results.get()
    return jsonify({'results':resp})                                                 # JSON ouput


@celery.task()                                                              # celery worker
def worker_process(payload):
    
    urls = payload['images']
    img_list = pipeline.load_imgs(urls)
    img_list_valid = [im for im in img_list if im['data'] is not None]
    
    try:
        module                                                              # if undefined,
    except NameError:
        module =  pipeline.build_module()                                   # build MXNet module thread safe for inference 
    
    pred = pipeline.process_images(img_list_valid, module).asnumpy()        # inference results 
    result = pipeline.make_results(img_list, pred)                          # classes and scores

    return result

if __name__ == '__main__':
    app.run(debug=True)
    

