FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 FLASK_HOST=0.0.0.0 MODELS_DIR=/app/models HF_HOME=/hf_cache

RUN pip install --no-cache-dir torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY Application_web ./Application_web
WORKDIR /app/Application_web
EXPOSE 5000
CMD ["python", "app.py"]