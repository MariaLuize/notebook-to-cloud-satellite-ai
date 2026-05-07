# base python image
# FROM python:3.10-slim
FROM tensorflow/tensorflow:2.16.1-gpu

# enviroment
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# GDAL instalacao
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*
# variáveis de ambiente do GDAL
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# working directory
WORKDIR /app

# copy requirements.txt and install requirements
COPY requirements.txt .
# pip a ignorar a versão do Ubuntu e compilar pro Python 3.11
RUN pip install --no-cache-dir --ignore-installed gdal==3.4.1
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
RUN chmod +x run.sh

ENTRYPOINT ["./run.sh"]
