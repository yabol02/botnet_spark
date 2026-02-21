FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# PYTHONPATH correcto (sin warning)
ENV PYTHONPATH=/app

RUN apt-get update && apt-get install -y \
    openjdk-11-jdk \
    wget \
    python3-pip \
    curl && \
    apt-get clean

WORKDIR /app

RUN pip3 install --upgrade pip

COPY docker/requirements.txt .
RUN pip3 install -r requirements.txt

# ✅ ahora sí existe
COPY functions ./functions

EXPOSE 8888

COPY docker/entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh

CMD ["/entrypoint.sh"]