FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY fun_asr_openai_proxy ./fun_asr_openai_proxy

EXPOSE 8000

CMD ["uv", "run", "--no-sync", "python", "-m", "fun_asr_openai_proxy"]
