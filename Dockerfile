# start with the official Airflow image (added from docker-compose.yaml)
# changing to latest stable version 2.10
FROM apache/airflow:2.10.2

COPY requirements.txt /requirements.txt

# Install the packages  - runs ONCE during build, not every startup 
RUN pip install --no-cache-dir -r /requirements.txt