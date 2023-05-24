FROM python:3.7.16

ARG APP_PATH=/app
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
WORKDIR $APP_PATH
COPY . $APP_PATH
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
EXPOSE 8000
# docker run -d -p 127.0.0.1:8000:8000 --gpus all -t tog_streamlit
CMD ["streamlit", "run", "main.py", "--server.port", "8000"]