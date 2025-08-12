# syntax=docker/dockerfile:1
FROM node:20-bookworm-slim

# System deps for OpenCV and tooling
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git ffmpeg \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create and activate Python virtualenv for pip installs (PEP 668 compliant)
RUN python3 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -U pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Node/TS sources
COPY package.json tsconfig.json ./
COPY src ./src
COPY scripts ./scripts

# Build TypeScript
RUN npm install && npm run build

# Ensure Node CLI uses the venv python inside the container
ENV VENV_PY=/opt/venv/bin/python

# Default entrypoint runs the built Node CLI; pass -i/-o via `docker run ... -- <args>`
ENTRYPOINT ["node", "dist/cli.js"] 