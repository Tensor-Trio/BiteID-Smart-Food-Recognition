FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    BITEID_MODEL_PATH=/srv/models/best_transfer_model.keras

WORKDIR /srv/app

# Build context is the repository root so the src package can be shared.
COPY app/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /srv/src/
COPY app/app.py ./app.py
COPY app/sample_images/ ./sample_images/

# Model weights are mounted at runtime rather than baked in, because they are
# generated artifacts and not version controlled. See docker-compose.yml.
RUN mkdir -p /srv/models

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
