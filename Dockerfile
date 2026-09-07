FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.4 /uv /bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY shim.py ./
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health',timeout=3)"
CMD ["uvicorn", "shim:app", "--host", "0.0.0.0", "--port", "8000"]
