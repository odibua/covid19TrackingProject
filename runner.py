from celery import Celery
import os

app = Celery('runner', backend='rpc://', broker='pyamqp://guest@localhost//')

@app.task
def add(x, y):
    return x + y