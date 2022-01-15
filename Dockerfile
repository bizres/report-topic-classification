FROM ubuntu:20.04 AS base

WORKDIR /usr/src/app

COPY Pipfile* ./

# Ubuntu 20.04 comes with Python 3.8 installed -> Install 3.9 version
RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get install -y python3.9 python3-pip

RUN ["pip3", "install", "pipenv"]

RUN ["pipenv", "install", "--system", "--deploy", "--ignore-pipfile"]

COPY . .

FROM base AS dev

RUN echo "Docker container for report topic classification is running..."