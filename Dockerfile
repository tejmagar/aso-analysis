# CPU-only torch: the GPU wheel is ~2.5GB and this project never uses it.
FROM python:3.12-slim AS base

# The database is Postgres now and its address comes from the environment;
# /data keeps only the embedding cache, which is a rebuildable artefact.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    ASO_EMBED_CACHE=/data/emb

# git: the Play Store reader installs from a repo, not PyPI.
RUN apt-get update && apt-get install -y --no-install-recommends git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so a code change does not re-download torch.
COPY pyproject.toml README.md ./
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install "fastapi" "uvicorn[standard]" \
    && pip install "google-play-api-unofficial @ git+https://github.com/tejmagar/google-play-api-unofficial" \
    && pip install sentence-transformers "psycopg[binary]"

COPY aso/ ./aso/
COPY schema_pg.sql ./
COPY models/ ./models/
RUN pip install -e . --no-deps

# Bake the sentence encoder into the image. Downloading it on first request
# would make the first analysis of a fresh container time out.
RUN python -c "from sentence_transformers import SentenceTransformer as S; \
    S('sentence-transformers/all-MiniLM-L6-v2')"

RUN mkdir -p /data && useradd -m -u 1000 aso && chown -R aso /data /models /app
USER aso
VOLUME ["/data"]
EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8765/health || exit 1

CMD ["python", "-m", "uvicorn", "aso.api:app", \
     "--host", "0.0.0.0", "--port", "8765", \
     "--log-level", "warning", "--timeout-keep-alive", "65"]
