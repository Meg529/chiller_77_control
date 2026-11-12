FROM python:3.11-slim

# 即時輸出 log
ENV PYTHONUNBUFFERED=1

# lightgbm 需要的系統函式庫
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN pip install --no-cache-dir uv

COPY . .

RUN uv sync --frozen --no-dev

CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
