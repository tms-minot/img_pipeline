import sys, os
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/../src"))
from flask import Response
import app
import pytest
import conftest
from flask import url_for
from celery import shared_task

def test_json_response(client):
    res = client.get(url_for('post_data'))
    assert res.json['hello'] == 42

@shared_task                                                                # Workaround, Celery docs broken
def hard_worker():
    return 'hi'
    
def test_create_task(celery_app, celery_worker):                                                          
    assert hard_worker.apply_async().get() == 'hi'

    