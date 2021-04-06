import os

#NOTE: These are publicly exposed access and secret keys
#S3_END_POINT = os.getenv('S3_END_POINT', 'http://192.168.1.205:9000')
#S3_END_POINT = os.getenv('S3_END_POINT', 'https://s3.openshift-storage.svc')
S3_END_POINT = os.getenv('S3_END_POINT', 'https://s3-openshift-storage.apps.zero.massopen.cloud/')
S3_ACCESS_ID = os.getenv('AWS_ACCESS_KEY_ID', 'v3FnruQ78kfeULDjejUB')
S3_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY', 'kJiDiHXncLJOXbaL7Zeb5Ok+gkLt9sWIa1rWAJa0')

BASE_IMAGE = 'docker.io/pytorch/pytorch:1.0-cuda10.0-cudnn7-runtime'
BUCKET_NAME = 'opf-datacatalog'