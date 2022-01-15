# Report Topic Classification

A natural language processing module for the sustainability report topic classification.

## Running application with Docker

To run the application with Docker, install the [Docker Engine](https://docs.docker.com/engine/install/) and [Docker Compose](https://docs.docker.com/compose/install/) on your development machine.

We currently use the official Ubuntu base image ([ubuntu:20.04](https://hub.docker.com/_/ubuntu)) and install the **Python 3.9** version.

#### Development build

```powershell
# Run the application in dev mode in Docker
# Use --build option to rebuild the container after any changes on environment (ie. installing modules)
# Use --detach option to run the container in background; it will running until you stop it
$:> docker-compose up [--build] [--detach]

# Ensure, that the container is running
$:> docker ps
CONTAINER ID   IMAGE                         COMMAND                  CREATED         STATUS         PORTS     NAMES
5a4a0e0956d3   bizres-topic-classifier:dev   "/bin/sh -c 'tail -f…"   7 minutes ago   Up 5 minutes             bizres-topic-classifier-dev

# SSH access into the container
$:> docker exec -it bizres-topic-classifier-dev /bin/bash
```