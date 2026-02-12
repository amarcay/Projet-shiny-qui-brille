FROM python:3.13-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project definition first (layer caching)
COPY pyproject.toml ./

# Install dependencies (no editable install, just deps)
RUN uv pip install --system --no-cache -r pyproject.toml

# Copy entire project (models, data, reports, src)
COPY . .

# Set best model version from registry
ENV MODEL_VERSION=v2

# Expose both ports
EXPOSE 5000 8000

# Start both services
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

CMD ["/docker-entrypoint.sh"]
