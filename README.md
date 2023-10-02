# FaceShape API

This is an implementation of the `model_128_sgd_reg.h5` model to make predictions via API.


## REST API

The REST API was build on Flask.

To run, you must follow these steps:
 1. Install the requirements using `pip install -r requirements.txt`
 2. Install the web requirements using `pip install -r requirements-web.txt`
 3. Run the aplication using `flask run`

The Application should be running on a webserver.


## AWS Lambda

This implemenation was designed to run on AWS Lambda.
To upload this model to AWS, you must fallow these steps:
 1. Create an `ECR` repository on AWS
 2. Follow AWS steps to build and upload the `Dockerfile`
 3. Create an Lambda on AWS using this Dockefile on your `ECR` repository.


## The Model

This model (`model_128_sgd_reg.h5`) was created as part of a Advanced Degree.
If you want more informations about, feel free to contact me.
