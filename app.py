#!flask/bin/python
from flask import Flask, jsonify
import pipeline
from celery import Celery


app = Flask(__name__)
app.config.update(
    CELERY_BROKER_URL='ampq://localhost:6379',
    CELERY_RESULT_BACKEND='ampq://localhost:6379')
celery = make_celery(app)


@app.route('/api/infer', methods=['POST'])
def post_data(payload):
    urls = payload['images']
    results = worker_process(urls)
    return jsonify(results)


@celery.task()
def worker_process(urls):
    
    
    img_list = pipeline.load_imgs(urls)
    img_list_valid = [im for im in img_list if im['data'] is not None]
    
    try:
        module
    except NameError:
        module =  pipeline.build_module(batch_size)    
    
    pred = pipeline.process_images(img_list_valid, module).asnumpy()

    result = pipeline.make_results(img_list, pred, thr)

    return result

if __name__ == '__main__':
    app.run(debug=True)
    

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