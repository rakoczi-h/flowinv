FROM --platform=linux/amd64 python:3.10 

RUN apt-get update && \
    apt-get install -y vim

COPY requirements.txt /tmp/requirements.txt

RUN useradd -ms /bin/bash flow

USER flow

RUN python -m ensurepip && \
    pip3 install nvidia-pyindex --user && \ 
    pip3 install -r /tmp/requirements.txt --user

WORKDIR /home/flow/flowinv

COPY --chown=flow:flow flowinv /home/flow/flowinv

ENV PATH ${PATH}:/home/flow/.local/bin

ENTRYPOINT echo "Go with the flow \033[0;31m♥\033[0m" && /bin/bash
