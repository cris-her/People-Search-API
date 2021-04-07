FROM nvcr.io/nvidia/nemo:v1.0.0b1

EXPOSE 6006

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

CMD ["python3", "src/server.py"]