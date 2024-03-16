FROM python:3.10

WORKDIR /gog-task

COPY ./requirements.txt /gog-task/requirements.txt
COPY ./data /gog-task/data
COPY ./src /gog-task/src
COPY ./ai /gog-task/ai
COPY ./main.py /gog-task/main.py

RUN pip install --no-cache-dir --upgrade -r /gog-task/requirements.txt

CMD ["python", "-m", "main"]

