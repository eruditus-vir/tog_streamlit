FROM python:3.6.15

ARG APP_PATH=/app
ARG POETRY_VERSION=1.3.0

WORKDIR $APP_PATH
RUN apk add --no-cache \
        curl \
        gcc \
        libressl-dev \
        musl-dev \
        libffi-dev && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile=minimal && \
    source $HOME/.cargo/env && \
    pip install --no-cache-dir poetry==${POETRY_VERSION} && \
    apk del \
        curl \
        gcc \
        libressl-dev \
        musl-dev \
        libffi-dev
COPY . $APP_PATH
RUN poetry install --no-interaction --no-ansi -vvv
RUN poetry run pytest
EXPOSE 8000
CMD ["poetry","run","jupyter-notebook", "--ip=0.0.0.0", "--port=8000"]