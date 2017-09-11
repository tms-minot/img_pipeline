import sys, os
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/../src"))
from flask import Response
import app
import pytest

class t_response(Response):
    @property
    def json(self):
        return {'hello':42}

@pytest.fixture
def t_app():
    t_app = app.app
    t_app.response_class = t_response
    return t_app

