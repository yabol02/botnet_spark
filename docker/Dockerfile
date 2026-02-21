# Base image
FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# Install dependencies
RUN apt-get update && apt-get install -y \
    openjdk-11-jdk \
    wget \
    python3-pip \
    curl && \
    apt-get clean

# Install Kafka
WORKDIR /opt

# Uprade pip and install dependencies
RUN pip3 install --upgrade pip
COPY requirements.txt /requirements.txt
RUN pip3 install -r /requirements.txt

# Expose ports
EXPOSE 8888

RUN ls

# Add entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh

# Default command
CMD ["/entrypoint.sh"]