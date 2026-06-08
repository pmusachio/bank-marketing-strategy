FROM python:3.11-slim

WORKDIR /project

ENV PYTHONPATH=/project/src \
    PYTHONUNBUFFERED=1

COPY requirements.txt requirements-api.txt requirements-app.txt ./
RUN python -m pip install --no-cache-dir \
    -r requirements.txt \
    -r requirements-api.txt \
    -r requirements-app.txt

COPY . .

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8501 8000

ENTRYPOINT ["entrypoint.sh"]
CMD ["app"]
