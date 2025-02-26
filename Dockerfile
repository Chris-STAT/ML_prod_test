FROM python:3.8

WORKDIR /opt/ml/code

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python", "train.py"]
