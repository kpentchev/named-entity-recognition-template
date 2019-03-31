FROM python:3.6
WORKDIR /app
COPY requirements.txt /app
RUN pip install -r ./requirements.txt
COPY *.py /app/
RUN mkdir -p /app/model/
COPY current_model.zip /app/models/
CMD ["python", "webApp.py"]~