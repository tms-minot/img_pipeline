#!flask/bin/python
from flask import Flask, jsonify
import pipeline
from celery import Celery


app = Flask(__name__)
app.config.update(
    CELERY_BROKER_URL='redis://localhost:6379',
    CELERY_RESULT_BACKEND='redis://localhost:6379')
celery = make_celery(app)


model = pipeline.build_model()


@app.route('/api/infer', methods=['POST'])
def post_data(payload):
    results = worker_process(payload)
    return jsonify(payload)


@celery.task()
def worker_process(payload):
    urls = payload['images']
    img_list = pipeline.load_imgs(urls)
    



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