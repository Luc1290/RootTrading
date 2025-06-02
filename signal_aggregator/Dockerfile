FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

COPY src/ ./src/

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["python", "-u", "src/main.py"]