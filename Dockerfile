FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 T2S_BACKEND=offline

COPY pyproject.toml README.md ./
COPY t2s ./t2s
COPY eval ./eval
RUN pip install --no-cache-dir ".[api]" \
 && python -m t2s.data \
 && useradd --create-home app && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "t2s.api:app", "--host", "0.0.0.0", "--port", "8000"]
