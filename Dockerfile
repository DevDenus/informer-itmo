FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    git wget curl build-essential \
    gcc g++ cmake && \
    rm -rf /var/lib/apt/lists/*
    
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x start.sh

CMD ["./start.sh"]
