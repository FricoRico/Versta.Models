FROM nvidia/cuda:13.0.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    build-essential cmake git ninja-build python3 python3-pip

RUN pip install packaging==25.0

COPY requirements-cuda.txt /tmp/
RUN pip install -r /tmp/requirements-cuda.txt

ENV MAX_JOBS=1
COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt --use-deprecated legacy-resolver

WORKDIR /workspace
COPY versta/ ./versta/
COPY .env ./.env

ENV PYTHONPATH=/workspace:$PYTHONPATH

ENTRYPOINT ["/usr/bin/env", "bash", "-c", "exec \"$@\"", "--"]
