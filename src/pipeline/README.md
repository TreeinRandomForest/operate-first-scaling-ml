# Hyperparameter grid search using pipelines

* config.py: has coordinates for s3 bucket. Note this bucket is public. Generally credentials should NOT be stored here. Read them using environment variables.

* utils.py: functions from reading from and writing to S3 buckets.

* hyperparam_grid_search_pipeline.py: run this to generate a YAML file that defines the pipeline architecture (DAG).