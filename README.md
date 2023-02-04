# TOG Streamlit Demo

## Current Implementation

A streamlit app that perform object detection and TOG adversarial attack based on https://github.com/git-disl/TOG

Currently, it's a single page app that can run on CPU-only computer.

This is also deployable with SSL and NGINX using https://github.com/SteveLTN/https-portal and docker-compose.

## Future Improvement

1. There's a better way to deploy this which is to make object detection as an API which can be hosted on something like
   SageMaker Endpoint for serverless inference. (this will be done on another repo)
    1. Streamlit app can be run as a standalone application on a free-tier EC2.
    2. The inference will be way faster because Sagemaker endpoint will only be billed for the processing time and not
       idle time.

## Known issues
1. Loading of models can cause error with multiple convd initilized or non-unique name
2. images uploaded from phone is rotated. 

