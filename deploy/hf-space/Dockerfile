# Hugging Face Space (Docker SDK, free CPU). This folder is the whole Space repo:
# the code is downloaded from GitHub at pinned commits, so the Space holds no copy of it.
# Secrets (Space settings): OPENAI_API_KEY, LLM_BASE_URL, LLM_MODEL.
FROM python:3.13-slim

ARG COPILOT_REF=855de26d12f816f1b6470c164da70fb5859cfb87
ARG RAG_REF=0ab14d2759c08fab2617d6c2f4b070fac9e415a1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    DATA_DIR=/app/data \
    COMPANY_NAME=TechNomad \
    LLM_PROVIDER=openai \
    LLM_EFFORT=low \
    LLM_MAX_TOKENS=1500 \
    RATE_LIMIT_PER_HOUR=20 \
    DEMO_KB_DIR=/opt/support-copilot/demo/kb \
    DEMO_TICKETS_FILE=/opt/support-copilot/demo/shop_tickets.json

WORKDIR /app

# CPU-only torch first: the default PyPI wheel on Linux pulls several GB of CUDA libraries.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# The RAG engine (not on PyPI) and the copilot itself, from GitHub.
RUN pip install "https://github.com/erbolka25/intelligent-document-search/archive/${RAG_REF}.tar.gz"
ADD https://github.com/erbolka25/support-copilot/archive/${COPILOT_REF}.tar.gz /tmp/copilot.tar.gz
RUN mkdir -p /opt/support-copilot \
    && tar -xzf /tmp/copilot.tar.gz -C /opt/support-copilot --strip-components=1 \
    && rm /tmp/copilot.tar.gz \
    && pip install /opt/support-copilot

# Spaces run the container as uid 1000.
RUN useradd --create-home --uid 1000 user && mkdir -p /app/data /app/.cache && chown -R user /app
USER user

# Bake the embedding model into the image: no download on every cold start.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')"

EXPOSE 7860
# One worker: the per-IP limiter lives in process memory.
CMD ["uvicorn", "support_copilot.main:app", "--host", "0.0.0.0", "--port", "7860"]
