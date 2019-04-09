FROM python:3.6
WORKDIR /app
COPY requirements.txt /app
RUN pip install -r ./requirements.txt
RUN mkdir -p /app/model/
COPY current_model.zip /app/models/
COPY *.py /app/
CMD ["gunicorn", "-b", "0.0.0.0:5000", "--threads", "2", "webApp:app"]