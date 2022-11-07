FROM python:3.7

ENV HOME /root
COPY requirements.txt $HOME
RUN apt-get update
RUN pip install --upgrade pip
WORKDIR $HOME
RUN python -m pip install -r requirements.txt
