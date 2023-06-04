# Dockerfile
FROM --platform=linux/amd64 ubuntu:22.04

# install python 
RUN apt-get update && apt-get install -y python3.10 python3-pip python3.10-venv

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# install pipx
RUN python3 -m pip install --user pipx

# add pipx to path
ENV PATH=/root/.local/bin:$PATH


# install poetry
RUN pipx install poetry

# configure poetry
RUN poetry config virtualenvs.create false

# Install any needed packages specified in pyproject.toml
RUN poetry install

CMD ["python", "mnist_draw_classifier/main.py"]
