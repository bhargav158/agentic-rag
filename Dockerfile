FROM python:3.11-slim

# git is needed to download the repositories users paste in
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

ENV CHROMA_PATH=/data/chroma_db \
    REPOS_DIR=/data/repos \
    PYTHONUNBUFFERED=1
RUN mkdir -p /data

EXPOSE 8000
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}"]
