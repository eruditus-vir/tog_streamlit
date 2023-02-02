FROM python:3.7.16

ARG APP_PATH=/app

WORKDIR $APP_PATH
COPY . $APP_PATH
RUN mkdir $APP_PATH/model_weights
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["streamlit", "run", "main.py", "--server.port", "8000"]