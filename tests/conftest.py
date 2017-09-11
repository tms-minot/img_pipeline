import sys, os
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))
from flask import Response
import app
import pytest

class t_response(Response):
    '''Implements custom deserialization method for response objects.'''
    @property
    def json(self):
        return {'hello':42}

@pytest.fixture
def t_app():
    t_app = app.app
    t_app.response_class = t_response
    return t_app

def test_json_response(client):
    res = client.get(url_for('api.infer'))
    assert res.json['hello'] == 42