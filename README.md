# TOG Streamlit Demo

## Current Implementation

A streamlit app that perform object detection and TOG adversarial attack based on https://github.com/git-disl/TOG

Currently, it's a single page app that can run on CPU-only computer.

## Future Improvement

1. There's a better way to deploy this which is to make object detection as an API which can be hosted on something like
   SageMaker Endpoint for serverless inference.
    1. Streamlit app can be run as a standalone application on a free-tier EC2.
    2. The inference will be way faster because Sagemaker endpoint will only be billed for the processing time and not
       idle time.

