FROM python:3.9-slim-bullseye

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && apt-get install -y gcc python3-dev \
    && apt-get install -y curl \
    && apt-get purge -y --auto-remove

RUN curl https://sdk.cloud.google.com > install.sh \
    && bash install.sh --disable-prompts

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONBUFFERED=1 PIP_NO_CACHE_DIR=1
ENV PATH $PATH:/root/google-cloud-sdk/bin

WORKDIR /opt/abf

COPY CHANGELOG.md LICENSE README.md batch.sh setup.py sandbox.ipynb ./
COPY src src

RUN pip install --upgrade pip && \
    pip install -e '.'

EXPOSE 8888

CMD jupyter-lab --ip 0.0.0.0 --no-browser --allow-root