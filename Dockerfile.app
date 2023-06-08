# Dockerfile
FROM --platform=linux/amd64 ubuntu:22.04

# install python 
RUN apt-get update && apt-get install -y python3.10 python3-pip python3.10-venv


# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# install poetry
RUN python3.10 -m pip install poetry

# configure poetry
RUN poetry config virtualenvs.create false


# Install any needed packages specified in pyproject.toml
RUN poetry install

CMD ["poetry", "run", "python3.10", "mnist_draw_classifier/main.py"]
