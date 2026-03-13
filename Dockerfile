FROM python:3.11-slim

WORKDIR /home

RUN apt-get update && apt-get upgrade

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python", "main.py"]