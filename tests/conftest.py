import sys, os
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/../src"))
from flask import Response
import app as flask_app
import pytest

class t_response(Response):
    @property
    def json(self):
        return {'hello':42}

@pytest.fixture
def app():
    app = flask_app.api
    app.response_class = t_response
    return app

@pytest.fixture(scope='session')
def celery_config():
    return {
        'broker_url': 'amqp://',
        'result_backend': 'amqp://'
    }