from celery import Celery
import os

app = Celery()
app.config_from_object('celeryconfig')

@app.task
def add(x, y):
    print("Hello")
    return x + y