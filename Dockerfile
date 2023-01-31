FROM python:3.7.16

ARG APP_PATH=/app

WORKDIR $APP_PATH
COPY . $APP_PATH
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["poetry","run","jupyter-notebook", "--ip=0.0.0.0", "--port=8000"]