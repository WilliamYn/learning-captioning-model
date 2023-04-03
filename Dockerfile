FROM python:3.8-slim-buster

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

WORKDIR /app
COPY flask_captioning.py /app

COPY requirements.txt requirements.txt
COPY learning_model /app/learning_model

# Install other requirements
RUN pip install -r requirements.txt
RUN python -m nltk.downloader punkt && python -m nltk.downloader stopwords

EXPOSE 80

CMD ["python", "flask_captioning.py"]