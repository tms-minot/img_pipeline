import sys, os
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/../src"))
from flask import Response
import app
import pytest


def test_json_response(client):
    res = client.get(url_for('api.infer'))
    assert res.json['hello'] == 42