#!flask/bin/python
from flask import Flask, request, jsonify
import pipeline
from celery import Celery
from celery.signals import worker_init

                                                              # Glob var
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

def make_api():
    
    flask_app = Flask(__name__)                                                 # http app with rabbitmq broker
    flask_app.config.update(
        CELERY_BROKER_URL='amqp://guest:guest@localhost:5672/',
        CELERY_RESULT_BACKEND='amqp://guest:guest@localhost:5672/')
        
    @flask_app.route('/api/infer', methods=['POST'])                            # API entry point decorator
    def post_data():
        payload = request.get_json(force=True)                                  # JSON input
        results = hard_worker.apply_async((payload,))
        resp = results.get()
        return jsonify({'results':resp})
        
    celery = make_celery(flask_app)
    @celery.task()                                                              # celery worker
    def hard_worker(payload):
        
        urls = payload['images']
        img_list = pipeline.load_imgs(urls)
        img_list_valid = [im for im in img_list if im['data'] is not None]
        
        global module
        if module is None:
            module =  pipeline.build_module()
        
        pred = pipeline.process_images(img_list_valid, module).asnumpy()        # inference results 
        result = pipeline.make_results(img_list, pred)                          # classes and scores
    
        return result
    return flask_app, celery

module = None                                                                   # MXNet engine global var
flask_app, celery = make_api()
                                                  

'''
@worker_init.connect                                                            # Only if inside worker
def inst_model(sender, **kwargs):
    global module 
    module = pipeline.build_module()                                            # build MXNet module thread safe for inference
'''

if __name__ == '__main__':
    flask_app.run(debug=True)


    

