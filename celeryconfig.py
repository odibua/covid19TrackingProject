from celery.schedules import crontab
broker_url = 'pyamqp://'
result_backend = 'rpc://'

task_serializer = 'json'
result_serializer = 'json'
accept_content = ['json']
timezone = 'US/Pacific'
beat_schedule = {
    'add-every-30-seconds': {
        'task': 'manager.main',
        'schedule': crontab(hour=7, minute=30),
    },
}