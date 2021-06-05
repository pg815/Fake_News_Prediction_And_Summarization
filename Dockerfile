FROM python:3.7.5
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
RUN python -m nltk.downloader punkt
RUN [ "python", "-c", "import nltk; nltk.download('all')" ]
ENTRYPOINT [ "python" ]
CMD [ "app.py" ]
